"""OpenAI-compatible endpoint backend (vLLM / SGLang / DashScope / Ollama)."""
from __future__ import annotations

import os
from typing import Sequence

from .base import Message


class OpenAICompatLLM:
    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY",
                 max_new_tokens: int = 512, temperature: float = 0.0):
        from openai import OpenAI

        self.name = model
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, "EMPTY"),
        )

    def chat(self, messages: Sequence[Message], *, max_new_tokens=None,
             temperature=None, stop=None) -> str:
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=list(messages),
            max_tokens=max_new_tokens or self.default_max_new_tokens,
            temperature=self.default_temperature if temperature is None else temperature,
            stop=list(stop) if stop else None,
        )
        return (resp.choices[0].message.content or "").strip()
