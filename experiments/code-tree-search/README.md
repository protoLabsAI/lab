# code-tree-search — execution-grounded tree search over a code model

A test-time prototype spun out of researching [`TsinghuaC3I/MARTI`](https://github.com/TsinghuaC3I/MARTI) / **MARS²** ([arXiv 2602.07848](https://arxiv.org/abs/2602.07848)), which does *learned* multi-agent tree search for code generation via RL (8×80G+ to train).

**What we can't do:** RL-train Ornith with MARS² — it's already maximally RL'd by DeepReinforce, and the rig is 2 cards, not 8×80G.

**What we can do (this):** take the *test-time* half of the idea and **ground it**. Where MARS² steers the search with a learned value model, we steer it with the [`code_exec`](../../evals/graders/code_exec.py) grader we already built — every candidate is **run against the hidden tests**, so the search follows real pass/fail, not a critic's guess. No training, fits our hardware, and it directly reuses the execution verifier from the eval work.

## How it works

```
greedy1 : one sample, no search              (the pass@1 baseline)
bestof  : k independent samples, keep best   (search, no refinement)
tree    : beam search — sample k, score by execution; expand the top-B partial
          solutions by showing the model its code + the FAILING tests/errors and
          asking for a fix; keep top-B; repeat for R rounds (or until solved).
```

The lever is the refinement step: **execution feedback (which exact asserts failed, with errors) flows back into the next generation.** That's the grounding MARS² gets from RL, available to us at inference for free.

```bash
# gateway key read from ../../evals/.env
~/dev/lab/.venv/bin/python search.py --model protolabs/smart --modes greedy1,tree \
    --tasks ../../evals/tasks/coding/hard.yaml --k 4 --beam 2 --rounds 3
```

`solved` = all hidden tests pass. Reported alongside `gens` (generations spent) so any lift is always read against its cost.

## Results (2026-06-30, hard.yaml, k=3 beam=2 rounds=2)

```
model               greedy1 (pass@1)      tree-search           gens
protolabs/smart     6/6  mean 1.000       6/6  mean 1.000       6 → 18
  (Ornith-35B)      └─ ceiling: one-shots everything, search adds only cost
gemma4-12b          5/6  mean 0.967       6/6  mean 1.000       6 → 18
  (weaker, non-Qwen) └─ search RECOVERED the one task it failed one-shot (hard_calc)
```

**Read:** the harness works and is model-agnostic. Where a model has headroom, execution-grounded search lifts it (gemma 5/6 → 6/6); where it's at ceiling, search is pure cost (Ornith 6/6 → 6/6 at 3× gens). Confirms the fusion-thread finding from the other side: **the lift is real but only exists where the base model is actually failing.** Ornith-specific lift therefore needs coding tasks genuinely beyond its one-shot reach — the same novel/adversarial-task bottleneck the `code_exec` suite surfaced. (gemma's lift came from best-of-k resampling; the refinement loop is implemented but gemma's headroom was too thin to exercise it — a harder task set would.)

## Wiring into protoAgent — a `coder` subagent (design)

[protoAgent](https://github.com/protoLabsAI/protoAgent) (LangGraph A2A) already talks to our gateway (`graph/llm.py`) and has a subagent system (`graph/subagents/config.py`, DeerFlow `task()` delegation) + git-URL-installable plugins. The fit: package this as a plugin that registers a **`coder` subagent**. The main agent delegates a *verifiable* coding subtask; the subagent runs the search loop and returns a **test-verified** solution — the main agent sees only the result, not the rollouts (progressive disclosure).

The one hard dependency is the **verifier**: caller-supplied tests → `code_exec` runner, or the `acp` delegate (protoCLI's real sandbox) for open-ended repo work. **No oracle = no grounding** → it degrades to best-of-k. This shines on *verifiable* coding.

## Combining with fusion — same shape, better selector

Fusion (self-MoA) and tree-search are the *same fan-out* — the only difference is the selector:

```
fusion:       fan out N  → JUDGE selects (blind)
tree-search:  fan out N  → EXECUTION selects → refine (grounded)
```

So don't bolt them together — let **fusion be the generator and execution be the selector** ("fusion proposes, tests dispose"). This fixes fusion's two measured limits: it can't tool-call (irrelevant — it's only generating), and its blind judge (replaced by the test oracle; on code-with-tests you never needed a judge — just run them). Realized as a **difficulty-gated escalation** inside the subagent:

```
1. greedy (1-shot smart)            cheap, solves most
2. best-of-k + execution-select     headroom recovery (gemma 5/6 → 6/6)
3. tree-search: refine on failures  grounded fix loop
4. fusion candidates + execute      hardest: richer proposals, oracle-selected
```

Each rung fires only when the cheaper one fails its tests, so fusion's ~3× (× search width) is paid **only** on the genuinely hard, verifiable problems both techniques were built for.

## Status

Prototype. The harness is the durable piece (verifier-grounded test-time search, model-agnostic via the gateway). Next steps if pursued: real MCTS (UCB node selection) instead of beam; harder/novel task set with headroom for the 35B; and — the actual research direction — Ornith's self-scaffolding *specifying* the search, per [`experiments/game-rlvr/`](../game-rlvr/).
