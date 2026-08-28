"""Turn experiments/<name>/rows.jsonl into the proposal's tables.

    python -m raf.analyze experiments/ra_fm_qwen3b_dev
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import mean


def load(path: str):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(exp_dir: str) -> None:
    rows = load(f"{exp_dir}/rows.jsonl")

    # success by (method, budget)
    cell = defaultdict(list)
    fidelity = defaultdict(list)
    util_loss = defaultdict(float)
    tau_bad = defaultdict(int)
    for r in rows:
        key = (r["objective"], r["budget_frac"])
        cell[key].append(r["success"])
        f = r.get("forget")
        if f:
            util_loss[key] += f["est_utility_loss"]
            tau_bad[key] += int(f["tau_violation"])
            for pm in f["per_memory"]:
                if pm.get("deleted") and pm.get("recoverability") is not None:
                    fidelity[key].append(pm["recoverability"])

    budgets = sorted({b for _, b in cell})
    methods = sorted({m for m, _ in cell})

    oracle = mean(cell.get(("ra_fm", 1.0), [0])) if ("ra_fm", 1.0) in cell else None

    print(f"\n=== Task success rate (oracle/full memory = "
          f"{oracle:.3f}) ===" if oracle is not None else "=== Task success rate ===")
    hdr = "method".ljust(22) + "".join(f"b={b:<7}" for b in budgets)
    print(hdr)
    for m in methods:
        line = m.ljust(22)
        for b in budgets:
            vals = cell.get((m, b))
            line += (f"{mean(vals):<9.3f}" if vals else "-".ljust(9))
        print(line)

    print("\n=== Mean recoverability of DELETED memories "
          "(reconstruction accuracy proxy) ===")
    for m in methods:
        line = m.ljust(22)
        for b in budgets:
            vals = fidelity.get((m, b))
            line += (f"{mean(vals):<9.3f}" if vals else "-".ljust(9))
        print(line)

    print("\n=== Estimated cumulative utility loss  sum I_i*(1-R_i) ===")
    for m in methods:
        line = m.ljust(22)
        for b in budgets:
            line += f"{util_loss.get((m, b), 0.0):<9.3f}"
        print(line)

    print("\n=== tau-threshold violations (forced deletions below tau) ===")
    for m in methods:
        line = m.ljust(22)
        for b in budgets:
            line += f"{tau_bad.get((m, b), 0):<9d}"
        print(line)

    if oracle is not None:
        print("\n=== Deletion regret  (oracle_success - method_success) ===")
        for m in methods:
            line = m.ljust(22)
            for b in budgets:
                vals = cell.get((m, b))
                line += (f"{oracle - mean(vals):<9.3f}" if vals else "-".ljust(9))
            print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/ra_fm_qwen3b_dev")
