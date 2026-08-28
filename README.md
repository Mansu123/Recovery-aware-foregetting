# Recoverability-Aware Forgetting (RA-FM)

Reference implementation of the proposal *"Recoverability-Aware Forgetting: A
Principled Framework for Safe Memory Deletion in Long-Horizon AI Agents"*,
evaluated on **[AppWorld](https://appworld.dev)** with a **Qwen** agent.

> **Central thesis.** An agent should forget information based not only on how
> useful it is, but on whether it would stay *reconstructable* from the memories
> that remain.

---

## What is implemented

| Proposal element | Where |
|---|---|
| Memory store `M`, retrieval, eviction | [`raf/memory/store.py`](raf/memory/store.py) |
| Importance layer `I_i = f_importance(m_i)` | [`raf/memory/importance.py`](raf/memory/importance.py) |
| **Reconstruction / recoverability layer `R_i`** | [`raf/memory/recoverability.py`](raf/memory/recoverability.py) |
| Deletion risk `D_i = I_i^γ (1 − R_i)` + budget/τ objective (RA-FM, §6–7) | [`raf/memory/forgetter.py`](raf/memory/forgetter.py) |
| Baselines: FIFO, frequency, importance-only, redundancy, recoverability-only, random (§11, §13) | [`raf/memory/forgetter.py`](raf/memory/forgetter.py) |
| ReAct agent over AppWorld | [`raf/agent/react_agent.py`](raf/agent/react_agent.py) |
| Experiment grid: dataset × budget × method, with oracle & deletion-regret (§10, §12) | [`raf/appworld_runner.py`](raf/appworld_runner.py) |
| Metrics: success, reconstruction accuracy, memory efficiency, deletion regret | [`raf/metrics.py`](raf/metrics.py) |
| Table generator | [`raf/analyze.py`](raf/analyze.py) |
| Deterministic proof-of-concept (no GPU) | [`raf/selftest.py`](raf/selftest.py) |

---

## Install

```bash
pip install -r requirements.txt
appworld install
appworld download data          # creates ./data  (APPWORLD_ROOT defaults to cwd)
```

## Run

```bash
# 1. deterministic check that the recoverability layer works (seconds, no model)
python main.py unit

# 2. tiny end-to-end check: Qwen2.5-0.5B on 2 AppWorld tasks, budget 50%
python main.py smoke

# 3. full grid
python main.py run --config configs/qwen3b_dev.json \
    --budgets 0.1,0.25,0.5,0.75 \
    --methods ra_fm,importance_only,recoverability_only,fifo,frequency,redundancy,random

python -m raf.analyze experiments/ra_fm_qwen3b_dev
```

### Serving Qwen

- **Local (`--backend hf`)** — `transformers` on CPU / Apple MPS / CUDA. Fine for
  ≤0.5B on an 8 GB laptop; use a real GPU for Qwen3-3B/4B.
- **Endpoint (`--backend openai --base-url ...`)** — any OpenAI-compatible server
  (vLLM, SGLang, Ollama, DashScope). Recommended for the full benchmark.
- `reconstructor_model` can differ from the agent model — the reconstruction
  layer is a *frozen* call and never fine-tuned (RQ2).

---

## The reconstruction layer (RA-FM's core)

See the module docstring in
[`raf/memory/recoverability.py`](raf/memory/recoverability.py) for the full
vocabulary. In brief, for each candidate memory `m_i`:

```
reconstruction basis   C_i   = Retrieve_k(m_i, M \ {m_i})     # only surviving evidence
reconstructor          g     : frozen LLM,  r_i = g(C_i, cue_i)
fidelity               φ_i   = w_cos·cos_emb(m_i, r_i) + w_nli·entails(r_i ⊨ m_i)
recoverability         R_i   = E[φ_i]              # Monte-Carlo mean
irrecoverability             = 1 − R_i
deletion risk          D_i   = I_i^γ · (1 − R_i)
```

Three **cost tiers**, each a cheaper lower bound on the next; `estimate()` only
escalates for memories whose cheap redundancy proxy is in the ambiguous band:

| tier | method | cost |
|---|---|---|
| L0 | `ρ_i = max_{c∈C_i} cos_emb(m_i, c)` — pure embedding geometry | free |
| L1 | extractive: stitch `C_i`, score fidelity | 1 embed pass |
| L2 | generative: `g(C_i, cue_i)` + entailment judge | LLM calls |

**Forgetting rule (§7 step 4):** protect every `m_i` with `R_i < τ`; among the
rest delete lowest `D_i` first until under budget `B`. If the budget can only be
met by deleting protected items, RA-FM deletes the least-risky ones and logs a
`tau_violation` so Theorem 3's budget–information tradeoff stays auditable.

---

## Output layout

```
experiments/<experiment_name>/
├── config.json      # exact config used
├── summary.json     # per (method,budget): success, fidelity, util loss, regret
└── rows.jsonl       # per task: success, steps, memory size, full forget trace
```
