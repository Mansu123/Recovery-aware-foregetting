"""Importance layer:  I_i = f_importance(m_i)  in [0, 1].

``I_i`` is the *expected future utility* of a memory. RA-FM deliberately keeps
this separate from recoverability -- a memory can be important yet safe to drop
(redundant), or unimportant yet unsafe (irrecoverable).
"""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from .base_scoring import clip01

if TYPE_CHECKING:
    from .item import MemoryItem
    from .store import MemoryStore

_SPECIFIC = re.compile(
    r"\b(\d{4}-\d\d-\d\d|\d+\.\d+|\$\d+|id[:=]|password|token|@\w+\.\w+|"
    r"phone|address|api_key|access_token)\b", re.I)


class ImportanceModel:
    def __init__(self, cfg, llm=None):
        self.cfg = cfg
        self.llm = llm

    # ------------------------------------------------------------------ #
    def score(self, item: "MemoryItem", store: "MemoryStore") -> float:
        if self.cfg.importance == "uniform":
            return 0.5
        if self.cfg.importance == "llm" and self.llm is not None:
            return self._llm_score(item)
        return self._heuristic(item, store)

    # ------------------------------------------------------------------ #
    def _heuristic(self, item: "MemoryItem", store: "MemoryStore") -> float:
        now = store.write_step
        age = max(0, now - item.write_step)
        recency = 0.5 ** (age / max(1, self.cfg.imp_recency_halflife))

        access = 1.0 - math.exp(-item.access_count / 2.0)

        specificity = 1.0 if _SPECIFIC.search(item.text) else 0.35
        if item.kind in ("fact", "outcome", "plan"):
            specificity = max(specificity, 0.7)

        # memories explicitly tagged as tied to the supervisor's request
        task_link = 1.0 if item.meta.get("goal_relevant") else 0.4

        w = self.cfg
        parts = [
            (w.imp_recency_weight, recency),
            (w.imp_access_weight, access),
            (w.imp_specificity_weight, specificity),
            (w.imp_task_link_weight, task_link),
        ]
        total_w = sum(wt for wt, _ in parts) or 1.0
        raw = sum(wt * v for wt, v in parts) / total_w
        return clip01(raw)

    # ------------------------------------------------------------------ #
    def _llm_score(self, item: "MemoryItem") -> float:
        msg = [
            {"role": "system", "content":
             "Rate how useful this note is for completing future tasks for the "
             "same user. Reply with a single number 0-10."},
            {"role": "user", "content": item.text},
        ]
        out = self.llm.chat(msg, max_new_tokens=8, temperature=0.0)
        m = re.search(r"\d+(\.\d+)?", out)
        return clip01(float(m.group()) / 10.0) if m else 0.5
