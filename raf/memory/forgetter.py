r"""RA-FM eviction policy.

Deletion objective (proposal section 6):
    given budget B, choose retained set S, |S| <= B, that
      * maximises retained useful information, and
      * keeps every *deleted* memory reconstructable above threshold tau.

Concrete rule (proposal section 7, step 4):
    1. score importance  I_i          (importance.py)
    2. score recoverability R_i        (recoverability.py, the reconstruction layer)
    3. deletion risk  D_i = I_i ** gamma * (1 - R_i)
    4. PROTECT   {i : R_i < tau}                      -- never delete
       EVICTABLE {i : R_i >= tau}, ascending D_i      -- delete cheapest first
       stop when |S| <= B  (or evictable set exhausted)

If protecting everything above the risk floor still leaves |S| > B, RA-FM has
to break tau. It then deletes the least-risky protected items and records the
`tau_violation` so the budget-information tradeoff (Theorem 3) is auditable.

The class also implements the ablation baselines so every method shares one
code path and one logging format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random

from .base_scoring import clip01
from .importance import ImportanceModel
from .recoverability import RecoverabilityEstimator
from .embedder import cosine


@dataclass
class ForgetTrace:
    trigger_step: int
    objective: str
    size_before: int
    size_after: int
    budget: int
    deleted_ids: list[int] = field(default_factory=list)
    protected_ids: list[int] = field(default_factory=list)
    tau_violation: bool = False
    est_utility_loss: float = 0.0        # sum of I_i*(1-R_i) over deletions
    est_info_loss: float = 0.0           # sum of (1-R_i) over deletions
    per_memory: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class RaFmForgetter:
    def __init__(self, cfg, importance: ImportanceModel,
                 recover: RecoverabilityEstimator):
        self.cfg = cfg
        self.importance = importance
        self.recover = recover
        self.traces: list[ForgetTrace] = []

    # ------------------------------------------------------------------ #
    def maybe_sweep(self, store, *, force: bool = False) -> ForgetTrace | None:
        budget = store.budget(self.cfg.budget_frac)
        over = len(store) > budget
        cadence = (self.cfg.sweep_every and
                   store.write_step % self.cfg.sweep_every == 0)
        if not (force or over or (cadence and len(store) > budget)):
            return None
        if len(store) <= budget:
            return None
        return self.sweep(store, budget)

    # ------------------------------------------------------------------ #
    def sweep(self, store, budget: int) -> ForgetTrace:
        obj = self.cfg.objective
        items = store.items()
        n_delete = len(items) - budget

        if obj == "ra_fm":
            deleted, protected, rows, tau_bad = self._ra_fm(store, items, n_delete)
        elif obj == "importance_only":
            deleted, protected, rows, tau_bad = self._importance_only(
                store, items, n_delete)
        elif obj == "recoverability_only":
            deleted, protected, rows, tau_bad = self._recoverability_only(
                store, items, n_delete)
        elif obj == "redundancy":
            deleted, protected, rows, tau_bad = self._redundancy(
                store, items, n_delete)
        elif obj == "frequency":
            deleted = [m.id for m in sorted(items, key=lambda m: m.access_count)
                       [:n_delete]]
            protected, tau_bad = [], False
            rows = self._plain_rows(items, deleted)
        elif obj == "fifo":
            deleted = [m.id for m in sorted(items, key=lambda m: m.write_step)
                       [:n_delete]]
            protected, tau_bad = [], False
            rows = self._plain_rows(items, deleted)
        elif obj == "random":
            rng = random.Random(store.write_step)
            deleted = [m.id for m in rng.sample(items, n_delete)]
            protected, tau_bad = [], False
            rows = self._plain_rows(items, deleted)
        else:
            raise ValueError(obj)

        size_before = len(store)
        by_id = {m.id: m for m in items}
        removed = store.remove(deleted)

        # Evaluation fairness (proposal 12, "reconstruction accuracy"): score the
        # recoverability of EVERY method's deletions against the memory that
        # actually remains -- so importance/FIFO/etc. are judged on the same
        # information-loss scale as RA-FM.
        for m in removed:
            if m.importance is None:
                m.importance = self.importance.score(m, store)
            rep = self.recover._estimate_one(m, store)  # noqa: SLF001
            m.recoverability = rep.recoverability
            m.redundancy_proxy = rep.redundancy_proxy
            for row in rows:
                if row.get("id") == m.id:
                    row["recoverability"] = round(m.recoverability, 4)
                    row["redundancy_proxy"] = round(m.redundancy_proxy, 4)

        util_loss = sum(
            (by_id[i].importance or 0.0) * (1.0 - (by_id[i].recoverability or 0.0))
            for i in deleted if i in by_id)
        info_loss = sum(
            1.0 - (by_id[i].recoverability or 0.0)
            for i in deleted if i in by_id)

        trace = ForgetTrace(
            trigger_step=store.write_step, objective=obj,
            size_before=size_before, size_after=len(store), budget=budget,
            deleted_ids=list(deleted), protected_ids=list(protected),
            tau_violation=tau_bad, est_utility_loss=util_loss,
            est_info_loss=info_loss, per_memory=rows)
        self.traces.append(trace)
        return trace

    # ------------------------------------------------------------------ #
    @staticmethod
    def _plain_rows(items, deleted):
        d = set(deleted)
        return [{"id": m.id, "kind": m.kind, "deleted": m.id in d,
                 "text": m.text[:200]} for m in items]

    # ------------------------------------------------------------------ #
    # RA-FM
    # ------------------------------------------------------------------ #
    def _score_all(self, store, items):
        reports = self.recover.estimate(store, items)
        for m in items:
            m.importance = self.importance.score(m, store)
            rep = reports[m.id]
            m.recoverability = rep.recoverability
            m.redundancy_proxy = rep.redundancy_proxy
            m.deletion_risk = (m.importance ** self.cfg.gamma) * (1.0 - m.recoverability)
        return reports

    def _ra_fm(self, store, items, n_delete):
        reports = self._score_all(store, items)
        tau = self.cfg.tau

        evictable = sorted((m for m in items if m.recoverability >= tau),
                           key=lambda m: m.deletion_risk)
        protected = [m for m in items if m.recoverability < tau]

        deleted = [m.id for m in evictable[:n_delete]]
        tau_violation = False

        if len(deleted) < n_delete:
            # must dip into protected set -> delete least-risky protected first
            need = n_delete - len(deleted)
            fallback = sorted(protected, key=lambda m: m.deletion_risk)[:need]
            deleted += [m.id for m in fallback]
            tau_violation = True

        rows = []
        for m in items:
            r = reports[m.id]
            rows.append({
                "id": m.id, "kind": m.kind, "importance": round(m.importance, 4),
                "recoverability": round(m.recoverability, 4),
                "redundancy_proxy": round(m.redundancy_proxy, 4),
                "deletion_risk": round(m.deletion_risk, 4),
                "tier": r.tier, "self_recovery_gap": round(r.self_recovery_gap, 4),
                "deleted": m.id in deleted,
                "protected": m.recoverability < tau,
                "text": m.text[:200],
            })
        return deleted, [m.id for m in protected], rows, tau_violation

    # ------------------------------------------------------------------ #
    def _importance_only(self, store, items, n_delete):
        for m in items:
            m.importance = self.importance.score(m, store)
            m.recoverability = m.recoverability or 0.0
        ranked = sorted(items, key=lambda m: m.importance)
        deleted = [m.id for m in ranked[:n_delete]]
        rows = [{"id": m.id, "importance": round(m.importance, 4),
                 "deleted": m.id in deleted, "text": m.text[:200]} for m in items]
        return deleted, [], rows, False

    def _recoverability_only(self, store, items, n_delete):
        reports = self.recover.estimate(store, items)
        for m in items:
            m.recoverability = reports[m.id].recoverability
            m.redundancy_proxy = reports[m.id].redundancy_proxy
        # delete the MOST recoverable (safest to lose)
        ranked = sorted(items, key=lambda m: m.recoverability, reverse=True)
        protected = [m.id for m in items if m.recoverability < self.cfg.tau]
        deleted = [m.id for m in ranked if m.id not in set(protected)][:n_delete]
        tau_bad = len(deleted) < n_delete
        if tau_bad:
            deleted += [m.id for m in ranked
                        if m.id not in set(deleted)][:n_delete - len(deleted)]
        rows = [{"id": m.id, "recoverability": round(m.recoverability, 4),
                 "deleted": m.id in deleted, "text": m.text[:200]} for m in items]
        return deleted, protected, rows, tau_bad

    def _redundancy(self, store, items, n_delete):
        # similarity/redundancy compression: drop items most similar to a kept one
        kept_ids: set[int] = set()
        order = sorted(items, key=lambda m: m.write_step)
        deleted: list[int] = []
        embs = {m.id: m.embedding for m in items}
        for m in order:
            if len(deleted) >= n_delete:
                kept_ids.add(m.id)
                continue
            red = max((cosine(embs[m.id], embs[k]) for k in kept_ids), default=0.0)
            if red >= 0.88:
                deleted.append(m.id)
            else:
                kept_ids.add(m.id)
        # if not enough removed, drop highest-similarity remaining
        if len(deleted) < n_delete:
            rest = [m for m in items if m.id not in set(deleted)]
            rest = sorted(
                rest,
                key=lambda m: max((cosine(embs[m.id], embs[k])
                                   for k in kept_ids if k != m.id), default=0.0),
                reverse=True)
            deleted += [m.id for m in rest[:n_delete - len(deleted)]]
        rows = [{"id": m.id, "deleted": m.id in deleted, "text": m.text[:200]}
                for m in items]
        return deleted, [], rows, False
