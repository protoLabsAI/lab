# rlm — Plan

> **This document is superseded by [PROPOSAL.md](PROPOSAL.md) as of 2026-05-02.**
>
> PROPOSAL.md is the authoritative system + execution plan after the antagonistic
> review. The phased milestones below were the M0–M4 sketch from initial
> scaffolding; they're kept here for historical context but should not be acted
> on as fresh work. Active iteration log lives in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## (Historical) original plan

## M0: Skeleton (this commit)
- LangGraph DAG: `plan → execute → (FINAL? END : plan)` loop
- Local exec sandbox with `RLM` (sync) + `RLM_MAP` (asyncio.gather parallel)
- Trajectory schema + JSONL writer
- Smoke test against local vLLMs (no gateway, no Langfuse yet)
- Hard budget guards: max_steps, max_tokens, max_wall_ms, max_depth

**Exit:** smoke test passes; trajectory file lands on disk with full turn record.

## M1: Tracing + gateway
- Langfuse callback on the graph (per-node spans, cost rollup)
- Route leaf calls through gateway (`protolabs/fast`) instead of vLLM-direct, keep planner direct (latency-sensitive)
- Run on real long-context task (lab repo as context, "find every place we mention X")

**Exit:** Langfuse trace visible end-to-end; one real task answered correctly.

## M2: Benchmarks
- Reproduce LoCoDiff against `RLM(planner=27B, leaf=35B)`
- Synthesize OOLONG-shaped task (~130K tokens) over a structured corpus
- Cost / latency / accuracy numbers vs raw 27B and raw 35B baselines

**Exit:** RESULTS.md with numbers, BLOG.md draft begun (per lab cycle).

## M3: protoCLI integration
- `protocli --rlm <prompt>` flag wires through to graph for >100K-token contexts
- Sandbox graduates to subprocess (Docker optional) for user-exposed runs

## M4: SFT post-train
- Once trajectory corpus reaches ~5K successful runs, SFT Qwen3.5-9B base
- Compare `RLM-Qwen3.5-9B` to `RLM(GPT-5-mini)` on OOLONG / BrowseComp-Plus
- Publish to `protoLabsAI/RLM-Qwen3.5-9B` + blog on protolabs.studio

## Open design questions parked for now
- LangGraph `Send` for per-leaf graph-level visibility vs `RLM_MAP` asyncio fan-out (currently the latter)
- IPython kernel sandbox vs subprocess vs Docker — start with `exec()`, graduate when needed
- Cross-session prefix-cache hint propagation (vLLM `cache_salt`) for sub-RLM calls sharing slices
