"""Tiny shared helpers for the scoring layers."""
from __future__ import annotations


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
