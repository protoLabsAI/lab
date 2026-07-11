# Phase 3 expanded-suite baseline — capability ladder (2026-07-02)

First baseline on the Phase 3 suites (`evals/PHASE3_RESULTS.md`). Purpose: prove the
new deterministic suites **discriminate across the ≤9B ladder** (the lab's north
star), even though a 35B thinking model saturates bounded single-turn tasks.

## Setup

    35B   Ornith-1.0-35B-FP8   :8000 (prod replica A)   served-model-name local
    9B    Ornith-1.0-9B (bf16) :8003 (replica B stood down for this run)   local
    4B    Qwen3.5-4B (bf16)    :8003 (after 9B)          local

All runs thinking-on, single trial, deterministic graders only (no LLM judge),
**fixed 8192-token budget** (`--max-tokens 8192`). Suites: reasoning v2 (24),
code-exec v2 (8), structured-hard (6), safety-agency (8). Replica B restored to
production after the run.

**Why a fixed thinking budget:** at the default 32K, the 9B fails to terminate
its chain-of-thought on the hard reasoning tiers — it thinks to the token cap
(~6 min/task at 85 tok/s) without converging, making the sweep intractable AND
non-reproducible. An 8192-token budget bounds every task and makes
thinking-termination-within-budget an explicit, fair capability axis (a model
that can't reach the answer in 8K tokens of thinking is genuinely weaker at that
task). All three models run under the identical budget.

## Ladder — per-suite mean task_score (8192-token budget, thinking-on)

    suite (n)            35B       9B       4B
    ------------------  ------   ------   ------
    reasoning (24)       0.882    0.726    0.615
    code-exec (8)        0.616    0.391    0.383
    structured (6)       0.882    0.882    0.849
    safety (8)           1.000    1.000    1.000
    ------------------  ------   ------   ------
    OVERALL              0.845    0.750    0.711

35B = Ornith-1.0-35B-FP8, 9B = Ornith-1.0-9B, 4B = Qwen3.5-4B. Raw per-task
results in `{35b,9b,4b}_results.json`.

## Read

- **reasoning v2 is the discriminator.** Clean, monotone spread 0.882 > 0.726 >
  0.615 across the exact ladder the lab ships (≤9B). This is what the Phase 3
  expansion was for — a deterministic, contamination-resistant suite that
  separates the small models we actually deploy.
- **The 8192-token budget is the hidden axis, not problem difficulty.** With
  unlimited thinking the 35B scored 0.962 on this same reasoning suite and ~0.99
  on code-exec (`PHASE3_RESULTS.md`, first calibration). Bounding thinking to 8K
  tokens dropped the 35B to 0.882 / 0.616 and opened the whole ladder. A model
  that can't reach the answer in 8K tokens of reasoning is genuinely weaker at
  the task; unbounded, they all eventually get there and the eval saturates.
- **code-exec separates big-from-small but not 9B-from-4B** (0.616 vs 0.391 vs
  0.383) — the spec-delta problems are a struggle zone for all three (no ceiling),
  and both small models fail them similarly.
- **structured-hard is a weak discriminator** (0.882 / 0.882 / 0.849) — composed
  invariants don't separate the ladder; even the 4B does them. Keep as a
  regression floor, not a capability separator.
- **safety-under-agency is flat at 1.000** — the naked credential-ask framing is
  trivially refused by all three. Confirmed too easy; the real T28 leak needs the
  audit-report framing (echoing config as the ostensible job). Rework as v2.

**Bottom line:** the suite now discriminates the ≤9B class on reasoning (and
big-vs-small on code-exec). The lever that made every suite informative was the
fixed thinking budget, not harder problems — which is itself the Phase 3 finding
and reproduces the Agents-A1 "scale the horizon" thesis: capable models saturate
per-turn difficulty given unlimited thinking; bound the budget (or extend the
horizon) and they separate.

## Provenance

Suite files at their committed seed (reasoning/code-exec `--seed 20260702`).
Raw per-task results in `custom_results.json` per model dir under the run
scratchpad; summary numbers transcribed here. Methodology-change rule: rerun on
any suite/grader edit.
