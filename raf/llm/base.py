"""Minimal chat-LLM interface shared by the agent and the reconstructor."""
from __future__ import annotations

from typing import Protocol, TypedDict, Sequence


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


class LLM(Protocol):
    """A synchronous chat model."""

    name: str

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        ...
