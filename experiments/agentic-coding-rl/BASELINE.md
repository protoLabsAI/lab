# Phase-0 Baseline — Qwen3-Coder-Next-NVFP4 (2026-07-05)

Model-under-test: `GadflyII/Qwen3-Coder-Next-NVFP4` served :8005 (Marlin, 1 card, 181 tok/s).
Judge: live `:8000/local` (replica-a, Ornith-35B-NVFP4) — NOT ava:4000 (dead-:8003 round-robin trap).
Results dir: `evals/results/codernext_20260705_021749/`

| suite | score | note |
|---|---|---|
| coding (custom hard-v2) | **67%** (16/24 pass^1) | exec grader, partial credit — GUARDRAIL |
| function_call (--all-suites) | **94%** (51/54) | ext 90 / inproc 96 / untagged 100 — GUARDRAIL |
| claw (T02,04,06,08,12,26) | **0.485** mean | T08 .91 T06 .75 T02 .72 T04 .54 · T12 0 T26 0 — **TARGET** |

**Read:** strong coder + tool-caller, weak multi-turn agentic scaffolding (0.485 vs Ornith ~0.74).
Distill target = lift claw, hold coding/FC. Fails exactly on judgment/recovery tasks (T12/T26=0),
nails mechanical ones (T08) — the self-scaffolding gap the Ornith teacher fills.

Gotcha logged: coding runner default `--max-tokens 32768` == served ctx → 400; use 8192 or serve ≥64K.
