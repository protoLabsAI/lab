# LoCoDiff baseline — RLM(planner=protolabs/smart, leaf=protolabs/fast)

First numbers on `experiments/rlm/` against [LoCoDiff-bench](https://github.com/AbanteAI/LoCoDiff-bench)
(Mentat AI, MIT-licensed). 20 tasks stratified across the four token-count quartiles
(5 each). Planner = Qwen3.6-27B-FP8 (thinking, 262K), leaf = Qwen3.6-35B-A3B-FP8
heretic, both via the gateway. concurrency=2, max_steps=24, max_wall=600s,
max_tokens=200K, planner temperature=0.

Run date: 2026-05-02. Total wall: 4488s (~75 min).

## Headline

| Bucket | Tokens (approx) | Pass | Avg wall | Avg tokens | Avg leaf calls |
|--------|----------------:|-----:|---------:|-----------:|---------------:|
| Q1     | 1.9k–21k        | 4/5 (80%) | 281s | 79,943 | 0.00 |
| Q2     | 21k–36k         | 2/5 (40%) | 465s | 140,365 | 0.00 |
| Q3     | 36k–60k         | 0/5 (0%)  | 596s | 141,352 | 0.00 |
| Q4     | 60k–98k         | 2/5 (40%) | 433s | 166,966 | 0.00 |
| **Overall** | — | **8/20 (40%)** | **449s** | **132,157** | **0.00** |

## What didn't work — and the one finding that matters

**Zero leaf calls across all 20 tasks.** The planner never used `RLM` or
`RLM_MAP` — not once. Every task was attempted as pure-Python single-shot
(write a diff parser, run it, debug it, output the result). This means we
benchmarked *the planner's ability to author a correct git-diff applier in
one Python script*, not the recursive-decomposition pattern the paper claims
for the win.

That fully explains the cliff at Q3:

- A diff parser robust enough to handle 50k+ tokens of multi-commit, multi-hunk,
  binary-mode, merge-conflict-resolved git history is genuinely hard to write
  correctly in one go.
- The planner spends 14–22 steps iterating on its parser, each iteration burning
  3–10k tokens, hits `max_steps=24` or `max_tokens=200k`, terminates without
  ever stepping back to "let me delegate per-commit work to the leaf in parallel".

The Q4 surprise (40% pass, beating Q3's 0%) is sample noise: two of the Q4
files (`defaultExternalContentHandlers.ts`, `field_index_base.rs`) had
predominantly context-line commits with few hunks, so the parser converged
quickly even on a long history.

## Failure modes by reason

| Reason | Count | Pattern |
|---|---:|---|
| `max_steps` (>= 24) | 9 | Iterating on diff parser; each pass adds another edge case |
| `error` (planner emitted neither code nor FINAL) | 3 | Bigger files: planner gives up mid-thought, emits prose only |

`max_tokens` was never the binding constraint, but the heaviest run hit 219k
of 200k cap (`mutable_numeric_index.rs` Q2, `editblock_coder.py` Q4). One of
the `error` cases (`args.zig` Q3) terminated cleanly at 8 steps / 40k tokens —
worth a per-trajectory look.

## Wall-clock distribution

- Fastest pass: 55s (`babel.config.js`, Q1, 7 steps)
- Slowest pass: 373s (`vite.config.ts`, Q1, 16 steps)
- Slowest fail: 782s (`Inspector.zig`, Q3, max_steps; wall budget overshoot
  because step-budget check fires every step, but a single multi-minute
  planner turn can blow through the wall budget before the next check)

This means our `max_wall_seconds` budget is soft: it's checked between
super-steps, not enforced as a hard kill. For real production use we'd need
a wall-clock kill via asyncio cancellation.

## What the paper claims, vs ours

| | RLM(GPT-5) (paper) | RLM(27B-thinking + 35B-A3B) (ours) |
|---|---|---|
| Q1-equivalent | not reported | 80% |
| Q4 (75k+ tokens) | 1-shot, "programmatic processing" | 40% |
| Vanilla baseline (Q4) | <10% (GPT-5 alone) | not yet measured |

Two things missing for a fair compare:
1. **Vanilla planner-only baseline** — feed the prompt straight to
   `protolabs/smart` (27B-FP8, 262K context) with no RLM scaffold and measure
   the cliff. The paper's "<10% on Q4" is for GPT-5 alone; we should measure
   ours.
2. **Trained-RLM comparison** — paper's RLM-Qwen3-8B post-train hits +28.3%
   over its base. Our planner is off-the-shelf. M4 in PLAN.md.

## Knobs we'd try next (in order of expected leverage)

1. **Prompt nudge for `RLM_MAP`** — the system prompt mentions the tool but
   doesn't tell the planner *when* to use it. Add a worked example for "apply
   N diffs in parallel chunks." This is the single biggest lever — it'd flip
   the experiment from "planner-as-Python-author" to "planner-as-orchestrator".
2. **Bigger budgets** — `max_steps=50`, `max_tokens=400k`, `max_wall=600`. Some
   Q3 failures were converging slowly; another 10 steps might've sealed them.
   (Schema defaults already updated to 50 / 600.)
3. **Stronger sandbox primitives** — make `unidiff` available in the REPL so
   the planner doesn't have to rewrite a diff parser. Trade-off: this is RLM
   moving toward "RAG-with-tools," away from the paper's pure orchestration
   thesis. Worth doing anyway because the goal is *winning the benchmark*.
4. **Hard wall-clock kill** — current `max_wall` is soft. asyncio cancellation
   in the planner-call await would cap real cost.

## Reproducibility

- Dataset: `git clone https://github.com/AbanteAI/LoCoDiff-bench /tmp/`
- Config: M0 commit + `max_steps=24, max_wall=600, max_tokens=200_000, temp=0`
- Run: `uv run python experiments/rlm/eval/run_locodiff.py --n 20 --concurrency 2 --strategy stratified --max-steps 24 --max-wall 600`
- Raw results: `experiments/rlm/results/locodiff-1777747137.jsonl`
- Trajectories: `/mnt/data/training/rlm-trajectories/trajectories-2026-05-02.jsonl`

## Honest call

The scaffold works (24/27 unit tests, end-to-end gateway routing, trajectories
captured for SFT). The accuracy story is "fine on small, broken on medium,
zero on large — because the planner doesn't know it's allowed to delegate."
Before drawing conclusions about the *recursive* part of Recursive Language
Models, we need a run where the planner actually recurses. That's the next
experiment, not a finding from this one.
