# PARKED — rlm

Parked 2026-05-22. No consumer (protoCLI/coding integration is out of scope; ORBIS retired). See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

Implementation of [Zhang/Kraska/Khattab 2025 — Recursive Language Models](https://arxiv.org/abs/2512.24601). Planner (Qwen3.6-27B-FP8 thinking, 262K) drives a sandboxed Python REPL holding context as a variable; recurses via `RLM(subquery, slice)` / `RLM_MAP(queries, slices)` to a leaf model (Qwen3.6-35B-A3B-FP8 heretic). LangGraph orchestration, Langfuse tracing, JSONL trajectory export for SFT.

## The finding that matters (RESULTS.md)

LoCoDiff-bench, 20 tasks across token quartiles, run 2026-05-02:

| Bucket | Pass | Avg leaf calls |
|---|---|---|
| Q1 (1.9k–21k) | 4/5 | 0.00 |
| Q2 (21k–36k) | 2/5 | 0.00 |
| Q3 (36k–60k) | 0/5 | 0.00 |
| Q4 (60k–98k) | 2/5 | 0.00 |
| **Overall** | **8/20 (40%)** | **0.00** |

**Zero leaf calls across all 20 tasks.** Planner never used `RLM` / `RLM_MAP` — every task was attempted single-shot in the REPL. We benchmarked the planner's ability to write a git-diff parser in one Python script, not the recursive-decomposition pattern.

## Why parked

The negative result is more interesting than a positive would have been. **This is the breakdown.** Write it up: *"We implemented Recursive LMs from the paper. The planner refused to recurse."* That post serves the brand better than fixing the prompt to force recursion would.

## How to resume

Don't resume the agent. Resume long enough to write the blog post — `RESULTS.md` is already 80% of the draft. Trajectories at `/mnt/data/training/rlm-trajectories/` are preserved for a possible "what would the planner have learned if it had recursed" SFT follow-up, if a consumer ever appears.

Gateway creds live in `~/.proto/.env`; smoke at `scripts/smoke.py`.
