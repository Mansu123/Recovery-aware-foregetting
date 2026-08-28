"""Deterministic sanity checks for the memory + recoverability + forgetter
stack. No GPU, no network -- uses a tiny stub reconstructor.

Checks the central claim of the proposal:
  * a redundant memory (its content is repeated elsewhere) scores HIGH
    recoverability and gets deleted;
  * a unique memory (a one-off research idea) scores LOW recoverability and
    is PROTECTED even though its importance is modest.
"""
from __future__ import annotations

import re

from .config import MemoryConfig, ForgetConfig
from .memory.store import MemoryStore
from .memory.importance import ImportanceModel
from .memory.recoverability import RecoverabilityEstimator
from .memory.forgetter import RaFmForgetter


class StubReconstructor:
    """A 'reconstructor' that just concatenates the basis it is given.
    Good enough to exercise fidelity scoring deterministically: if the deleted
    fact is genuinely present in the basis, the concatenation contains it."""

    name = "stub"

    def chat(self, messages, *, max_new_tokens=None, temperature=None, stop=None):
        user = messages[-1]["content"]
        sys = messages[0]["content"]
        if "REFERENCE" in user and "CANDIDATE" in user:
            ref = re.search(r"REFERENCE:\n(.*?)\n\nCANDIDATE:\n(.*)", user, re.S)
            if not ref:
                return "0.0"
            a = set(ref.group(1).lower().split())
            b = set(ref.group(2).lower().split())
            j = len(a & b) / max(1, len(a | b))
            return f"{1.0 if j > 0.6 else 0.5 if j > 0.3 else 0.0}"
        # reconstruction request: echo the remaining notes
        notes = re.findall(r"\[\d+\]\s*(.+)", user)
        return " ".join(notes)


def run_selftest() -> None:
    mcfg = MemoryConfig(embedder="hash", basis_k=4, proxy_low=0.15,
                        proxy_high=0.70, w_cos=0.5, w_nli=0.5,
                        importance="heuristic", imp_recency_halflife=4)
    tau = 0.60
    store = MemoryStore(mcfg)

    # redundant cluster: the user's coffee preference, stated many ways
    coffee = [
        store.add("The user prefers oat milk in their coffee.", kind="fact"),
        store.add("User asked again for oat milk, no sugar, in the latte.", kind="fact"),
        store.add("Reminder: user always orders coffee with oat milk.", kind="fact"),
        store.add("User: please use oat milk for my coffee.", kind="fact"),
    ]
    coffee_ids = {m.id for m in coffee}
    redundant = coffee[-1]

    # unique, irrecoverable: a one-off research idea with no echo anywhere
    unique = store.add(
        "Research idea: gate memory eviction on a rate-distortion style "
        "reconstruction bound computed from surviving neighbours only.",
        kind="plan", meta={"goal_relevant": False})

    # filler so there is something to delete under budget
    for i in range(4):
        store.add(f"Unrelated log line number {i}: opened the maps app.",
                  kind="observation")

    importance = ImportanceModel(mcfg)
    recover = RecoverabilityEstimator(mcfg, StubReconstructor())

    reports = recover.estimate(store)
    r_red = reports[redundant.id].recoverability
    r_uni = reports[unique.id].recoverability
    print(f"recoverability(redundant coffee fact) = {r_red:.3f}")
    print(f"recoverability(unique research idea)  = {r_uni:.3f}")
    assert r_red > r_uni, "redundant memory must be more recoverable than unique"
    assert r_uni < tau, "unique research idea should fall below tau"
    assert r_red >= tau, "redundant coffee fact should sit above tau"

    fcfg = ForgetConfig(objective="ra_fm", budget_frac=0.5, tau=tau,
                        sweep_every=0)
    forgetter = RaFmForgetter(fcfg, importance, recover)
    budget = store.budget(0.5)
    trace = forgetter.sweep(store, budget)

    kept = {m.id for m in store.items()}
    print("deleted ids:", trace.deleted_ids)
    print("protected ids:", trace.protected_ids)
    assert unique.id in kept, "RA-FM deleted an irrecoverable memory!"
    assert unique.id in trace.protected_ids
    deleted_coffee = coffee_ids & set(trace.deleted_ids)
    kept_coffee = coffee_ids & kept
    assert deleted_coffee, "RA-FM kept every copy of a fully redundant fact"
    assert kept_coffee, "RA-FM deleted the coffee fact entirely (info lost)"

    # ablation: an OLD, rarely-accessed unique idea looks low-importance, so
    # importance-only evicts it -- while RA-FM protects it on recoverability.
    def build_store2():
        s = MemoryStore(mcfg)
        uid = s.add(unique.text, kind="plan",
                    meta={"goal_relevant": False}).id
        for _ in range(3):
            s.add("The user prefers oat milk in their coffee.", kind="fact")
        for i in range(6):
            s.add(f"Recent task step {i}: fetched the order status from Amazon.",
                  kind="observation")
        return s, uid

    s2, uid2 = build_store2()
    t2 = RaFmForgetter(
        ForgetConfig(objective="importance_only", budget_frac=0.5, tau=tau,
                     sweep_every=0),
        importance, recover).sweep(s2, s2.budget(0.5))
    imp_dropped = uid2 in t2.deleted_ids

    s3, uid3 = build_store2()
    t3 = RaFmForgetter(
        ForgetConfig(objective="ra_fm", budget_frac=0.5, tau=tau, sweep_every=0),
        importance, recover).sweep(s3, s3.budget(0.5))
    rafm_kept = uid3 not in t3.deleted_ids

    print(f"\nimportance-only evicted the old unique idea? {imp_dropped}")
    print(f"RA-FM protected the old unique idea?        {rafm_kept}")
    assert imp_dropped and rafm_kept, (
        "expected importance-only to drop the unique idea and RA-FM to keep it")

    print("\nSELFTEST PASSED - recoverability layer separates redundant vs unique;"
          "\nRA-FM protects the irrecoverable memory where importance-only loses it.")
