"""The agent's episodic memory: append on write, retrieve top-k on read."""
from __future__ import annotations

from typing import Sequence
import numpy as np

from .embedder import build_embedder, cosine
from .item import MemoryItem


class MemoryStore:
    def __init__(self, cfg, embedder=None):
        self.cfg = cfg
        self._embedder = embedder or build_embedder(cfg)
        self._items: list[MemoryItem] = []
        self._next_id = 0
        self.write_step = 0
        self.peak_size = 0
        # history for the budget-information tradeoff study (Theorem 3)
        self.size_trace: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ #
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._embedder.encode(list(texts))

    # ------------------------------------------------------------------ #
    def add(self, text: str, *, kind: str = "observation", task_id: str = "",
            meta: dict | None = None) -> MemoryItem:
        text = text.strip()
        if not text:
            return None  # type: ignore[return-value]
        self.write_step += 1
        item = MemoryItem(
            id=self._next_id, text=text, kind=kind, task_id=task_id,
            write_step=self.write_step, last_access_step=self.write_step,
            embedding=self.embed([text])[0], meta=dict(meta or {}))
        self._next_id += 1
        self._items.append(item)
        self.peak_size = max(self.peak_size, len(self._items))
        self.size_trace.append((self.write_step, len(self._items)))
        return item

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int | None = None) -> list[MemoryItem]:
        if not self._items:
            return []
        k = k or self.cfg.retrieve_k
        q = self.embed([query])[0]
        ranked = sorted(self._items, key=lambda m: cosine(q, m.embedding),
                        reverse=True)[:k]
        for m in ranked:
            m.touch(self.write_step)
        return ranked

    # ------------------------------------------------------------------ #
    def remove(self, ids: Sequence[int]) -> list[MemoryItem]:
        drop = set(ids)
        removed = [m for m in self._items if m.id in drop]
        self._items = [m for m in self._items if m.id not in drop]
        self.size_trace.append((self.write_step, len(self._items)))
        return removed

    # ------------------------------------------------------------------ #
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def budget(self, frac: float) -> int:
        return max(1, int(round(frac * max(self.peak_size, len(self._items)))))

    def context_block(self, query: str, k: int | None = None) -> str:
        hits = self.retrieve(query, k)
        if not hits:
            return "(memory is empty)"
        return "\n".join(f"- [{m.kind}] {m.text}" for m in hits)
