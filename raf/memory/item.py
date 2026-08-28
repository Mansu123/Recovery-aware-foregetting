from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class MemoryItem:
    """One episodic memory the agent wrote during a task.

    ``text`` is the natural-language content that must survive deletion in
    reconstructable form. Everything else is bookkeeping used by the importance
    and recoverability layers.
    """

    id: int
    text: str
    kind: str = "observation"          # observation | action | fact | plan | outcome
    task_id: str = ""
    write_step: int = 0                 # global memory-write counter at creation
    last_access_step: int = 0
    access_count: int = 0
    embedding: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # cached scores (filled by the forgetter, kept for logging / ablations)
    importance: float | None = None
    recoverability: float | None = None
    redundancy_proxy: float | None = None
    deletion_risk: float | None = None

    def touch(self, step: int) -> None:
        self.last_access_step = step
        self.access_count += 1

    def to_row(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "task_id": self.task_id,
            "write_step": self.write_step, "access_count": self.access_count,
            "importance": self.importance, "recoverability": self.recoverability,
            "redundancy_proxy": self.redundancy_proxy,
            "deletion_risk": self.deletion_risk,
            "text": self.text[:280],
        }
