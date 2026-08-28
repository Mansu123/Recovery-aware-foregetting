from __future__ import annotations

from ..config import LLMConfig
from .base import LLM


def build_llm(cfg: LLMConfig, *, role: str = "agent") -> LLM:
    """role: "agent" -> cfg.model ; "reconstructor" -> cfg.reconstructor()."""
    model = cfg.model if role == "agent" else cfg.reconstructor()

    if cfg.backend == "openai":
        from .openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(
            model, base_url=cfg.base_url, api_key_env=cfg.api_key_env,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)

    from .hf_local import HFLocalLLM
    return HFLocalLLM(
        model, device=cfg.device, dtype=cfg.dtype,
        max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
        load_in_4bit=cfg.load_in_4bit)
