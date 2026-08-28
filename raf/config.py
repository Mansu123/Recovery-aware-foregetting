"""Configuration objects for RA-FM experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal
import json
import os

# --------------------------------------------------------------------------- #
# Model / backend                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class LLMConfig:
    """Which model backs both the *agent* and the *reconstructor*.

    The reconstructor (recoverability layer) can be a different, cheaper model
    than the agent; set ``reconstructor_model`` to override.
    """

    backend: Literal["hf", "openai"] = "hf"

    # Agent policy model.
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    # Reconstructor model (recoverability layer). None -> reuse ``model``.
    reconstructor_model: str | None = None

    # hf backend
    device: str = "auto"           # "auto" | "cpu" | "mps" | "cuda"
    dtype: str = "auto"            # "auto" | "float16" | "bfloat16" | "float32"
    max_new_tokens: int = 512
    temperature: float = 0.0
    load_in_4bit: bool = False     # requires bitsandbytes + CUDA

    # openai-compatible backend (vLLM / SGLang / DashScope / Ollama)
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"

    def reconstructor(self) -> str:
        return self.reconstructor_model or self.model


# --------------------------------------------------------------------------- #
# Memory + recoverability                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class MemoryConfig:
    """Retrieval + scoring hyper-parameters for the memory store."""

    embedder: Literal["hf", "hash"] = "hash"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 512           # only used by the hashing embedder
    retrieve_k: int = 6            # neighbours fed to the agent at inference

    # ---- recoverability (reconstruction) layer ----
    basis_k: int = 5               # size of the reconstruction basis C_i
    recon_samples: int = 1         # Monte-Carlo reconstructions per memory
    recon_temperature: float = 0.7 # >0 only matters when recon_samples > 1
    # tier gating: skip the expensive generative tier when the cheap redundancy
    # proxy is already decisive.
    proxy_low: float = 0.30        # proxy below this -> treat as irrecoverable
    proxy_high: float = 0.92       # proxy above this -> treat as recoverable
    # fidelity = w_cos * cosine + w_nli * entailment
    w_cos: float = 0.5
    w_nli: float = 0.5

    # ---- importance layer ----
    importance: Literal["llm", "heuristic", "uniform"] = "heuristic"
    # weights for the heuristic importance model
    imp_recency_halflife: int = 40   # in memory-write steps
    imp_recency_weight: float = 0.30
    imp_access_weight: float = 0.25
    imp_specificity_weight: float = 0.25
    imp_task_link_weight: float = 0.20


@dataclass
class ForgetConfig:
    """The RA-FM deletion objective."""

    # keep at most ``budget_frac`` of peak memory size
    budget_frac: float = 0.25
    # protect any memory whose recoverability is below tau
    tau: float = 0.65
    # deletion risk  D_i = importance_i ** gamma * (1 - R_i)
    gamma: float = 1.0
    # ablation switch
    objective: Literal["ra_fm", "importance_only", "recoverability_only",
                       "fifo", "frequency", "redundancy", "random"] = "ra_fm"
    # run the forgetter every N memory writes (0 -> only when over budget)
    sweep_every: int = 8


@dataclass
class RunConfig:
    dataset: Literal["train", "dev", "test_normal", "test_challenge"] = "train"
    experiment_name: str = "ra_fm_qwen3b"
    max_interactions: int = 40
    limit: int | None = None          # cap number of tasks (smoke tests)
    seed: int = 0
    output_dir: str = "experiments"


@dataclass
class RafConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    forget: ForgetConfig = field(default_factory=ForgetConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ------------------------------------------------------------------ #
    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "RafConfig":
        with open(path) as fh:
            raw = json.load(fh)
        return cls(
            llm=LLMConfig(**raw.get("llm", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            forget=ForgetConfig(**raw.get("forget", {})),
            run=RunConfig(**raw.get("run", {})),
        )

    @classmethod
    def smoke(cls) -> "RafConfig":
        """Tiny config that actually runs on an 8 GB laptop."""
        cfg = cls()
        cfg.llm.model = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.llm.max_new_tokens = 384
        cfg.memory.embedder = "hash"
        cfg.memory.recon_samples = 1
        cfg.run.dataset = "train"
        cfg.run.limit = 2
        cfg.run.max_interactions = 12
        cfg.run.experiment_name = "ra_fm_smoke"
        return cfg
