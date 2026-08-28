"""Evaluation metrics from proposal section 12."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class SessionMetrics:
    method: str
    budget_frac: float
    task_success: list[int] = field(default_factory=list)
    # reconstruction fidelity of the memories that were actually deleted
    deletion_fidelity: list[float] = field(default_factory=list)
    est_utility_loss: float = 0.0
    est_info_loss: float = 0.0
    tau_violations: int = 0
    peak_memory: int = 0
    final_memory: int = 0
    tokens_reconstruction: int = 0

    # ------------------------------------------------------------------ #
    def summary(self, oracle_success: list[int] | None = None) -> dict:
        sr = mean(self.task_success) if self.task_success else 0.0
        out = {
            "method": self.method,
            "budget_frac": self.budget_frac,
            "success_rate": round(sr, 4),
            "n_tasks": len(self.task_success),
            "mean_deletion_fidelity": round(
                mean(self.deletion_fidelity), 4) if self.deletion_fidelity else None,
            "est_utility_loss": round(self.est_utility_loss, 4),
            "est_info_loss": round(self.est_info_loss, 4),
            "tau_violations": self.tau_violations,
            "peak_memory": self.peak_memory,
            "final_memory": self.final_memory,
            # memory efficiency = useful task success retained per kept memory
            "memory_efficiency": round(
                sr / max(1, self.final_memory), 6),
        }
        if oracle_success is not None and oracle_success:
            # deletion regret: performance gap vs full (un-forgotten) memory
            out["deletion_regret"] = round(
                mean(oracle_success) - sr, 4)
        return out
