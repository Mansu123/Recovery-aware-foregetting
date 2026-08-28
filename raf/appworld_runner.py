"""Run the RA-FM experiment grid on AppWorld.

A "session sequence" = the group of AppWorld tasks that share a scenario prefix
(e.g. 82e2fac_1, 82e2fac_2, ...). The agent keeps ONE MemoryStore across the
whole sequence, so memories written in early tasks are what later tasks depend
on -- the long-horizon setting the proposal targets. The forgetter sweeps
between tasks (and mid-task on cadence), so tighter budgets force real deletions.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, OrderedDict

from .config import RafConfig
from .llm import build_llm
from .memory.store import MemoryStore
from .memory.importance import ImportanceModel
from .memory.recoverability import RecoverabilityEstimator
from .memory.forgetter import RaFmForgetter
from .agent.react_agent import ReactAgent
from .metrics import SessionMetrics


def _group_task_ids(task_ids: list[str]) -> "OrderedDict[str, list[str]]":
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    for tid in task_ids:
        prefix = tid.split("_")[0]
        groups.setdefault(prefix, []).append(tid)
    for k in groups:
        groups[k].sort(key=lambda t: int(t.split("_")[-1])
                       if t.split("_")[-1].isdigit() else 0)
    return groups


class ExperimentRunner:
    def __init__(self, cfg: RafConfig):
        self.cfg = cfg
        self.agent_llm = build_llm(cfg.llm, role="agent")
        self.recon_llm = (self.agent_llm if cfg.llm.reconstructor_model is None
                          else build_llm(cfg.llm, role="reconstructor"))

    # ------------------------------------------------------------------ #
    def _fresh_memory_stack(self):
        store = MemoryStore(self.cfg.memory)
        importance = ImportanceModel(
            self.cfg.memory,
            llm=self.agent_llm if self.cfg.memory.importance == "llm" else None)
        recover = RecoverabilityEstimator(self.cfg.memory, self.recon_llm)
        forgetter = RaFmForgetter(self.cfg.forget, importance, recover)
        return store, forgetter

    # ------------------------------------------------------------------ #
    def run_method(self, objective: str, budget_frac: float,
                   task_groups) -> tuple[SessionMetrics, list[dict]]:
        from appworld import AppWorld

        cfg = self.cfg
        cfg.forget.objective = objective
        cfg.forget.budget_frac = budget_frac

        metrics = SessionMetrics(method=objective, budget_frac=budget_frac)
        rows: list[dict] = []

        for prefix, tids in task_groups.items():
            store, forgetter = self._fresh_memory_stack()
            agent = ReactAgent(self.agent_llm, store,
                               max_interactions=cfg.run.max_interactions)

            for tid in tids:
                with AppWorld(task_id=tid,
                              experiment_name=f"{cfg.run.experiment_name}"
                                              f"__{objective}__b{int(budget_frac*100)}"
                              ) as world:
                    res = agent.run(world)

                success = int(bool(res.eval and res.eval.get("success")))
                metrics.task_success.append(success)

                trace = forgetter.maybe_sweep(store)
                if trace is not None:
                    self._absorb_trace(trace, store, metrics)

                rows.append({
                    "group": prefix, "task_id": tid, "objective": objective,
                    "budget_frac": budget_frac, "success": success,
                    "steps": res.num_steps, "answer": res.answer,
                    "mem_recalled": res.memories_recalled,
                    "mem_written": res.memories_written,
                    "mem_size": len(store),
                    "forget": trace.as_dict() if trace else None,
                })

            # end-of-sequence forced sweep to the budget
            final_trace = forgetter.maybe_sweep(store, force=True)
            if final_trace is not None:
                self._absorb_trace(final_trace, store, metrics)
            metrics.peak_memory = max(metrics.peak_memory, store.peak_size)
            metrics.final_memory += len(store)

        return metrics, rows

    # ------------------------------------------------------------------ #
    @staticmethod
    def _absorb_trace(trace, store, metrics: SessionMetrics) -> None:
        metrics.est_utility_loss += trace.est_utility_loss
        metrics.est_info_loss += trace.est_info_loss
        metrics.tau_violations += int(trace.tau_violation)
        for row in trace.per_memory:
            if row.get("deleted") and row.get("recoverability") is not None:
                metrics.deletion_fidelity.append(row["recoverability"])

    # ------------------------------------------------------------------ #
    def run_grid(self, methods: list[str], budgets: list[float]) -> dict:
        from appworld import load_task_ids

        cfg = self.cfg
        task_ids = load_task_ids(cfg.run.dataset)
        if cfg.run.limit:
            task_ids = task_ids[: cfg.run.limit]
        groups = _group_task_ids(task_ids)

        out_dir = os.path.join(cfg.run.output_dir, cfg.run.experiment_name)
        os.makedirs(out_dir, exist_ok=True)
        cfg.to_json(os.path.join(out_dir, "config.json"))

        # oracle: full memory, no forgetting (budget 1.0, objective ra_fm but
        # never triggers a sweep because size <= budget)
        summaries: list[dict] = []
        all_rows: list[dict] = []

        oracle_metrics, oracle_rows = self.run_method("ra_fm", 1.0, groups)
        oracle_success = oracle_metrics.task_success
        summaries.append(oracle_metrics.summary())
        all_rows += oracle_rows

        for budget in budgets:
            for method in methods:
                m, rows = self.run_method(method, budget, groups)
                summaries.append(m.summary(oracle_success=oracle_success))
                all_rows += rows
                self._flush(out_dir, summaries, all_rows)

        self._flush(out_dir, summaries, all_rows)
        return {"summaries": summaries, "output_dir": out_dir}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _flush(out_dir: str, summaries, rows) -> None:
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summaries, fh, indent=2)
        with open(os.path.join(out_dir, "rows.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
