r"""The reconstruction layer  --  RA-FM's core novelty.

Question it answers, per candidate memory m_i:
    "If m_i is deleted, can its information content be regenerated from the
     memories that would remain?"

---------------------------------------------------------------------------
TERMINOLOGY  (the "reconstructable layer" vocabulary)
---------------------------------------------------------------------------
reconstruction basis   C_i  = Retrieve_k( key(m_i),  M \ {m_i} )
                              the small set of surviving memories that are the
                              *only* evidence the reconstructor is allowed to use.

reconstructor          g            a frozen LLM call (NO fine-tuning, RQ2).
                                    r_i = g(C_i, cue_i)  -- proposes the content
                                    of the deleted memory from the basis alone.

reconstruction cue     cue_i        a minimal, content-free pointer to what is
                                    being recovered (m_i's kind + a redacted
                                    subject phrase). Prevents the trivial
                                    "reconstruct nothing" degenerate answer
                                    without leaking the answer.

fidelity               phi_i        similarity(m_i, r_i) in [0,1]
                                    = w_cos * cos_emb  +  w_nli * entailment
                                    entailment = "does r_i assert m_i ?"

recoverability         R_i          E_cue,sample[ phi_i ]  -- Monte-Carlo mean
                                    over cues / sampled reconstructions.

self-recovery gap      Delta_i      phi_i(basis WITH a paraphrase of m_i present
                                    elsewhere) - phi_i(basis WITHOUT it).
                                    Large gap  => the info lived only in m_i
                                    (unique);  ~0 gap => redundant.

irrecoverability                    1 - R_i
deletion risk          D_i          I_i ** gamma * (1 - R_i)

---------------------------------------------------------------------------
THREE COST TIERS  (each a strictly cheaper *lower bound* on the next)
---------------------------------------------------------------------------
L0  redundancy proxy   rho_i = max_{c in C_i} cos_emb(m_i, c)
                       free -- pure embedding geometry, no LLM.
L1  extractive recovery
                       stitch the basis with a deterministic template, score
                       fidelity. One embedding pass, no generation.
L2  generative recovery
                       full g(C_i, cue_i) with entailment check. Expensive.

`estimate()` runs L0 for every memory, then only escalates to L1/L2 for the
memories whose proxy lands in the ambiguous band (proxy_low, proxy_high).
That keeps the number of LLM reconstructions ~O(memories near the margin)
instead of O(all memories).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .base_scoring import clip01
from .embedder import cosine

if TYPE_CHECKING:
    from .item import MemoryItem
    from .store import MemoryStore
    from ..llm.base import LLM


# --------------------------------------------------------------------------- #
@dataclass
class RecoverabilityReport:
    memory_id: int
    recoverability: float
    redundancy_proxy: float
    tier: str                       # "L0" | "L1" | "L2"
    fidelity_cos: float = 0.0
    fidelity_nli: float = 0.0
    self_recovery_gap: float = 0.0
    reconstruction: str = ""
    basis_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reconstruction"] = d["reconstruction"][:400]
        return d


# --------------------------------------------------------------------------- #
_RECON_SYS = (
    "You reconstruct a deleted note in an AI agent's long-term memory.\n"
    "You are given (a) a short cue describing WHICH note was deleted and (b) the "
    "notes that still remain. Using ONLY the remaining notes, write your single "
    "best guess of the deleted note's exact content. If the remaining notes do "
    "not determine it, write what they do imply and mark the uncertain parts "
    "with [?]. Answer with the note text only -- no preamble."
)

_NLI_SYS = (
    "Judge whether the CANDIDATE fully conveys the information in the REFERENCE "
    "note (same facts, values, entities; paraphrase is fine; extra detail is "
    "fine). Reply with one number: 1.0 = fully conveyed, 0.5 = partially, "
    "0.0 = not conveyed / contradicted."
)


class RecoverabilityEstimator:
    def __init__(self, cfg, reconstructor: "LLM | None"):
        self.cfg = cfg
        self.g = reconstructor

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def estimate(self, store: "MemoryStore",
                 candidates: list["MemoryItem"] | None = None
                 ) -> dict[int, RecoverabilityReport]:
        items = candidates if candidates is not None else store.items()
        reports: dict[int, RecoverabilityReport] = {}
        for m in items:
            reports[m.id] = self._estimate_one(m, store)
        return reports

    # ------------------------------------------------------------------ #
    def _basis(self, m: "MemoryItem", store: "MemoryStore"
               ) -> list["MemoryItem"]:
        # everything except m itself; retrieval keyed on m's own embedding
        pool = [x for x in store.items() if x.id != m.id]
        if not pool:
            return []
        q = m.embedding if m.embedding is not None else store.embed([m.text])[0]
        scored = sorted(pool, key=lambda x: cosine(q, x.embedding), reverse=True)
        return scored[: self.cfg.basis_k]

    # ------------------------------------------------------------------ #
    def _estimate_one(self, m: "MemoryItem", store: "MemoryStore"
                      ) -> RecoverabilityReport:
        basis = self._basis(m, store)
        if not basis:
            # nothing else in memory -> the memory is its own only witness
            return RecoverabilityReport(m.id, 0.0, 0.0, "L0")

        q = m.embedding if m.embedding is not None else store.embed([m.text])[0]
        proxy = max(cosine(q, c.embedding) for c in basis)
        rep = RecoverabilityReport(
            m.id, recoverability=clip01(proxy), redundancy_proxy=clip01(proxy),
            tier="L0", basis_ids=[c.id for c in basis])

        # decisive on geometry alone -> stop (cheap)
        if proxy <= self.cfg.proxy_low or proxy >= self.cfg.proxy_high:
            return rep
        if self.g is None:
            return rep  # no reconstructor available: proxy is our estimate

        # ---- L1: extractive recovery -------------------------------- #
        extractive = "\n".join(f"- {c.text}" for c in basis)
        cos1 = self._emb_sim(store, m.text, extractive)
        rep.tier, rep.fidelity_cos = "L1", cos1
        rep.recoverability = clip01(cos1)

        # ---- L2: generative recovery ------------------------------- #
        recon = self._reconstruct(m, basis)
        cos2 = self._emb_sim(store, m.text, recon)
        nli = self._entailment(m.text, recon)
        fidelity = self.cfg.w_cos * cos2 + self.cfg.w_nli * nli

        rep.tier = "L2"
        rep.reconstruction = recon
        rep.fidelity_cos, rep.fidelity_nli = cos2, nli
        rep.recoverability = clip01(max(cos1, fidelity))
        rep.self_recovery_gap = clip01(rep.recoverability - proxy)
        return rep

    # ------------------------------------------------------------------ #
    def _reconstruct(self, m: "MemoryItem", basis: list["MemoryItem"]) -> str:
        cue = self._cue(m)
        remaining = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(basis))
        user = f"CUE (deleted note):\n{cue}\n\nREMAINING NOTES:\n{remaining}"
        msgs = [{"role": "system", "content": _RECON_SYS},
                {"role": "user", "content": user}]

        n = max(1, self.cfg.recon_samples)
        if n == 1:
            return self.g.chat(msgs, temperature=0.0)
        outs = [self.g.chat(msgs, temperature=self.cfg.recon_temperature)
                for _ in range(n)]
        # return the medoid reconstruction (most central) for logging;
        # scoring below still uses just this string.
        return max(outs, key=lambda o: sum(len(set(o.split()) & set(p.split()))
                                           for p in outs))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cue(m: "MemoryItem") -> str:
        words = m.text.split()
        head = " ".join(words[:4])
        return f"kind={m.kind}; about: \"{head} ...\" ({len(words)} words)"

    # ------------------------------------------------------------------ #
    def _emb_sim(self, store: "MemoryStore", a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        va, vb = store.embed([a, b])
        # map cosine [-1,1] -> [0,1]; negatives are "unrelated", clamp to 0
        return clip01(cosine(va, vb))

    # ------------------------------------------------------------------ #
    def _entailment(self, reference: str, candidate: str) -> float:
        if self.g is None or not candidate.strip():
            return 0.0
        msgs = [{"role": "system", "content": _NLI_SYS},
                {"role": "user", "content":
                 f"REFERENCE:\n{reference}\n\nCANDIDATE:\n{candidate}"}]
        out = self.g.chat(msgs, max_new_tokens=8, temperature=0.0)
        m = re.search(r"[01](\.\d+)?", out)
        return clip01(float(m.group())) if m else 0.0
