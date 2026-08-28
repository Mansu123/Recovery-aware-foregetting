"""Sentence embeddings for retrieval + fidelity scoring.

Two backends:
  * "hf"   - a small sentence-transformer loaded with plain transformers
             (mean pooling + L2 norm). Best quality, ~90 MB.
  * "hash" - a deterministic hashing embedder over word / char n-grams.
             Zero extra weights, good enough for redundancy geometry, and the
             only sane choice on a memory-constrained laptop running Qwen too.
"""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    _STOP = frozenset(
        "a an the of to in on at for and or but is are was were be been being "
        "i you he she it we they my your his her its our their me him us them "
        "this that these those with as by from please again no not do did".split())

    def _feats(self, text: str) -> list[tuple[str, float]]:
        words = _TOKEN.findall(text.lower())
        content = [w for w in words if w not in self._STOP]
        feats: list[tuple[str, float]] = []
        # content words carry most of the signal
        for w in content:
            feats.append((f"w:{w}", 3.0))
            feats.append((f"s:{w[:5]}", 1.0))              # light stemming
        for a, b in zip(content, content[1:]):
            feats.append((f"b:{a}_{b}", 2.0))
        for w in words:                                    # keep function words faint
            if w in self._STOP:
                feats.append((f"w:{w}", 0.4))
        return feats or [("<empty>", 1.0)]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for f, wt in self._feats(text):
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[r, h % self.dim] += sign * wt
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class HFEmbedder:
    def __init__(self, model: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


def build_embedder(cfg):
    if cfg.embedder == "hf":
        try:
            return HFEmbedder(cfg.embedder_model)
        except Exception as exc:  # noqa: BLE001 - fall back rather than crash a run
            print(f"[embedder] HF load failed ({exc}); using hashing embedder")
    return HashingEmbedder(cfg.embed_dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.clip(a @ b, -1.0, 1.0))
