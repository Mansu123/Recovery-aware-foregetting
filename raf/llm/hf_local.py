"""Local Qwen via HuggingFace transformers (CPU / Apple MPS / CUDA)."""
from __future__ import annotations

import threading
from typing import Sequence

from .base import Message


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, device: str):
    import torch

    if dtype != "auto":
        return getattr(torch, dtype)
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class HFLocalLLM:
    """Wraps a chat model; caches by model id so agent + reconstructor that
    share a checkpoint only load weights once."""

    _CACHE: dict = {}
    _LOCK = threading.Lock()

    def __init__(self, model: str, *, device: str = "auto", dtype: str = "auto",
                 max_new_tokens: int = 512, temperature: float = 0.0,
                 load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        dev = _resolve_device(device)
        self.device = dev

        key = (model, dev, dtype, load_in_4bit)
        with HFLocalLLM._LOCK:
            if key not in HFLocalLLM._CACHE:
                tok = AutoTokenizer.from_pretrained(model)
                kwargs: dict = {"torch_dtype": _resolve_dtype(dtype, dev)}
                if load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True)
                    kwargs["device_map"] = "auto"
                mdl = AutoModelForCausalLM.from_pretrained(model, **kwargs)
                if not load_in_4bit:
                    mdl = mdl.to(dev)
                mdl.eval()
                HFLocalLLM._CACHE[key] = (tok, mdl)
            self.tokenizer, self.model = HFLocalLLM._CACHE[key]

    # ------------------------------------------------------------------ #
    def chat(self, messages: Sequence[Message], *, max_new_tokens=None,
             temperature=None, stop=None) -> str:
        import torch

        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        temperature = self.default_temperature if temperature is None else temperature

        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        text = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text.strip()
