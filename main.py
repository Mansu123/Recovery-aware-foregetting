"""Recoverability-Aware Forgetting (RA-FM) -- entry point.

Usage:
    python main.py smoke                     # tiny end-to-end check (2 tasks)
    python main.py unit                      # deterministic memory-layer test, no GPU
    python main.py run  [--config cfg.json]  # full experiment grid
    python main.py run  --dataset dev --model Qwen/Qwen2.5-3B-Instruct \
                        --budgets 0.1,0.25,0.5,0.75 \
                        --methods ra_fm,importance_only,recoverability_only,fifo,frequency,redundancy,random
"""
from __future__ import annotations

import argparse
import json
import sys

from raf.config import RafConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default=None,
                   choices=["train", "dev", "test_normal", "test_challenge"])
    p.add_argument("--model", default=None)
    p.add_argument("--reconstructor-model", default=None)
    p.add_argument("--backend", default=None, choices=["hf", "openai"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--embedder", default=None, choices=["hf", "hash"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-interactions", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--budgets", default="0.1,0.25,0.5,0.75")
    p.add_argument("--methods",
                   default="ra_fm,importance_only,recoverability_only,"
                           "fifo,frequency,redundancy,random")


def _cfg_from_args(a) -> RafConfig:
    cfg = RafConfig.from_json(a.config) if a.config else RafConfig()
    if a.dataset: cfg.run.dataset = a.dataset
    if a.model: cfg.llm.model = a.model
    if a.reconstructor_model: cfg.llm.reconstructor_model = a.reconstructor_model
    if a.backend: cfg.llm.backend = a.backend
    if a.base_url: cfg.llm.base_url = a.base_url
    if a.embedder: cfg.memory.embedder = a.embedder
    if a.limit is not None: cfg.run.limit = a.limit
    if a.max_interactions: cfg.run.max_interactions = a.max_interactions
    if a.tau is not None: cfg.forget.tau = a.tau
    if a.experiment_name: cfg.run.experiment_name = a.experiment_name
    return cfg


def cmd_run(a) -> None:
    from raf.appworld_runner import ExperimentRunner

    cfg = _cfg_from_args(a)
    budgets = [float(x) for x in a.budgets.split(",") if x]
    methods = [x for x in a.methods.split(",") if x]
    runner = ExperimentRunner(cfg)
    res = runner.run_grid(methods, budgets)
    print(json.dumps(res["summaries"], indent=2))
    print("\nwrote:", res["output_dir"])


def cmd_smoke(a) -> None:
    from raf.appworld_runner import ExperimentRunner

    cfg = RafConfig.smoke()
    if a.model:
        cfg.llm.model = a.model
    runner = ExperimentRunner(cfg)
    res = runner.run_grid(["ra_fm", "fifo"], [0.5])
    print(json.dumps(res["summaries"], indent=2))
    print("\nwrote:", res["output_dir"])


def cmd_unit(_a) -> None:
    from raf.selftest import run_selftest
    run_selftest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "smoke"):
        sp = sub.add_parser(name)
        _add_common(sp)
    sub.add_parser("unit")

    a = ap.parse_args()
    {"run": cmd_run, "smoke": cmd_smoke, "unit": cmd_unit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
