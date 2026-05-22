# rlm — Recursive Language Models on protoLabs

Implementation of [Zhang/Kraska/Khattab 2025 — Recursive Language Models](https://arxiv.org/abs/2512.24601) wired into the protoLabs stack.

A root LM never sees the long context. The context is loaded as a Python variable in a sandboxed REPL the LM drives via fenced code blocks. The LM `peek`s, `grep`s, partitions, and calls `RLM(subquery, slice)` / `RLM_MAP(queries, slices)` recursively. Final answer is emitted via `FINAL(...)` or `FINAL_VAR(name)`.

## Why this exists

- **Avoid context rot** on long-context tasks (codebase audits, multi-doc QA, log triage) without paying frontier-model pricing
- **Capture trajectories** as a fine-tuning corpus — every successful run becomes one SFT example for an eventual `RLM-Qwen3.5-9B` post-train
- **Slot into protoCLI** as a `--rlm` mode so long-context workloads route here instead of cramming into one window

## Stack

| Role | Model | Endpoint | Why |
|---|---|---|---|
| Planner (root) | Qwen3.6-27B-FP8 (thinking) | `:8000` `local` | 262K context, agentic-tuned, decides decomposition |
| Worker (leaf) | Qwen3.6-35B-A3B-FP8 heretic | `:8002` `local-fast` | 226 tok/s, parallel-friendly via `RLM_MAP` |
| Orchestration | LangGraph | in-process | State machine + Langfuse tracing |
| Sandbox | Local `exec()` per session | in-process | Day-0 simplicity; will graduate to ipykernel/Docker |
| Tracing | Langfuse (already wired) | `langfuse.protolabs` | Per-leaf cost + quality |
| Trajectories | JSONL | `/mnt/data/training/rlm-trajectories/` | SFT corpus for future post-train |

## Run

```bash
# Source gateway creds (LITELLM_API_KEY) — see ~/.proto/.env
set -a; source ~/.proto/.env; set +a
export GATEWAY_API_KEY="$LITELLM_API_KEY"

cd ~/dev/lab
uv run python experiments/rlm/scripts/smoke.py
```

Defaults route through the gateway (`http://ava:4000/v1`) so traces land in Langfuse.
Override via `GATEWAY_URL` env var or by passing a custom `RLMConfig`.

## Layout

```
rlm/
├── schema.py        # Pydantic Trajectory + Turn; LangGraph State TypedDict
├── sandbox.py       # Local exec sandbox with stdout capture + RLM/RLM_MAP injection
├── parser.py        # Fenced-block extraction, FINAL detection
├── llm.py           # Async OpenAI client wrappers for planner + leaf
├── prompts.py       # Root planner system prompt
├── graph.py         # LangGraph DAG + budget guards + Langfuse callback
└── trajectory.py    # JSONL persistence
scripts/smoke.py     # End-to-end toy run
tests/               # Unit tests for parser + sandbox
```

See [PLAN.md](PLAN.md) for milestones.
