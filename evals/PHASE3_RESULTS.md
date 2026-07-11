# Eval Phase 3 — expansion + calibration RESULTS (2026-07-02)

Internal, honest numbers. Goal: lower the baseline / restore discrimination after
the Agents-A1 challenger run showed models tying at the top of every suite. All
runs: Ornith-35B-FP8 daily driver (the standing replica), thinking-on, single trial.

## What was built

| suite | tasks | grader | LLM judge? | new? |
|---|---|---|---|---|
| reasoning v2 | 24 (6 families × 4) | match: number/regex/all_of | no | replaces 5-task v1 |
| code-exec v2 | 8 spec-delta | code_exec (hidden + property tests) | no | new |
| structured-hard | 6 composed-invariant | json_validate (assertion battery) | no | new grader + suite |
| safety-under-agency | 8 T28 attack vectors | match: none_of (leak variants) | no | new |
| long-horizon T201 | 1 (of 5 designed) | sandbox verify + weighted grader | no | new claw task |

New graders: `json_validate.py` (parse + assertion battery, partial credit),
`match.py` gained `none_of` (leak/refusal). FC grader gained an
`allowed_intermediate` whitelist so legitimate grounding calls (`current_time`
before a relative-date query — the gina_019/021 false-negatives) stop being
penalized as "extra."

## Calibration on Ornith-35B

    suite                 mean    struggle-zone tasks         verdict
    --------------------  ------  --------------------------  -----------------------------
    reasoning v2          0.962   1/24 (arith-medium 0.50)    STILL CEILINGS on 35B
    code-exec v2          ~0.99   0/8 (one 0.93)              spec-delta ≠ hard for 35B
    structured-hard       0.882   2/6 (ledger .62, fsm .67)   partially discriminates
    safety-under-agency   1.00    0/8                         framing too easy (naked ask)
    long-horizon T201     n/a     ref-solution validated 1.0  needs Docker calibration run

## The finding (this is the content)

**A 35B thinking model saturates every BOUNDED, SINGLE-TURN suite we can author —
regardless of grading rigor or de-memorization.** We removed the LLM judge, used
seeded contamination-clean generators, and cranked difficulty (30-operation
register traces, 3-stream interleaved sequences, 7-inhabitant knights & knaves,
relational-only zebra grids). The 35B still scored ≥0.9 on 21/24 reasoning tasks.
Spec-delta coding problems (memorized solution compiles and FAILS) — 35B solves
them anyway. Naked credential-leak asks — trivially refused.

Where discrimination DID appear, the pattern is consistent and tells us what to
build:
1. **Multi-value partial-credit answers** — arith-medium (report BOTH item counts,
   0.50), schedule (all_of over 6 assignments, 0.75–0.83). One slip costs a point;
   a single scalar answer the model either nails or doesn't.
2. **Exact multi-field computation** — structured-hard ledger (0.62) and
   state-machine (0.67): a JSON object where every field is an independent
   computed invariant.
3. **Long-horizon multi-turn** (T201, calibration pending) — the axis where errors
   compound across 20–50 tool calls.

**This reproduces the Agents-A1 "scale the horizon, not the parameters" thesis from
the eval side.** Per-turn difficulty saturates; horizon and answer-multiplicity are
what separate capable models. You cannot lower a strong model's ceiling with harder
puzzles — only with longer tasks and answers that have more independent ways to be
wrong.

## Ladder re-baseline (2026-07-02, fixed 8192-token budget)

Full writeup + raw results: `baselines/2026-07-02-phase3-ladder/`. Standing down
replica B freed GPU 1 for the 9B and 4B; restored to production after.

    suite (n)            35B       9B       4B
    reasoning (24)       0.882    0.726    0.615     ← monotone, strong
    code-exec (8)        0.616    0.391    0.383     ← 35B ≫ small
    structured (6)       0.882    0.882    0.849     ← flat, weak
    safety (8)           1.000    1.000    1.000     ← flat, rework
    OVERALL              0.845    0.750    0.711

**The budget is the discriminator, not the difficulty.** The SAME reasoning suite
scored 0.962 on the 35B with unlimited thinking (above) and 0.882 when thinking
was capped at 8192 tokens; code-exec went 0.99 → 0.616. Bounding the thinking
budget is what opened the ladder — a model that can't reach the answer in 8K
tokens of reasoning is genuinely weaker at that task, and unbounded they all
eventually converge and the eval saturates. This is the cleaner statement of the
"can't lower a strong model's ceiling with harder puzzles" finding: you lower it
by bounding the horizon (thinking budget), which is the Agents-A1 thesis exactly.

Practical suite verdicts from the ladder:
- **reasoning v2 — keep as the primary ≤9B discriminator.** Monotone spread.
- **code-exec v2 — keep; separates big-vs-small, struggle zone on all.**
- **structured-hard — weak (near-flat); demote to regression floor.**
- **safety-under-agency — flat at 1.0; rework under audit-report framing (v2).**

## Consequence for the acceptance rule

The FOCUS Phase 3 rule "35B must score 0.3–0.7" is wrong for bounded suites — it's
unachievable without making tasks absurd. Revised:
- **Bounded suites (reasoning, structured) discriminate across the ladder we
  actually ship (≤9B), not at 35B.** The lab's north star is on-device ≤9B; a
  reasoning suite that ceilings on 35B but spreads 4B<9B is doing its job. Prefer
  **multi-value partial-credit answers** so even the 35B leaves headroom.
- **Struggle-zone-on-35B is reserved for the axes that actually bite it:**
  long-horizon agentic (T2xx) and exact multi-field computation.

## Remaining work (task #9 + follow-ups)

- **Ladder runs (4B/9B):** confirm reasoning v2 / structured-hard spread the ≤9B
  models. Needs a GPU plan — both cards hold 35B replicas + Fish + embed; serve
  9B/4B off-gateway on a freed slot (as the 2026-06-27 baseline did on :8005),
  do NOT disturb production replicas.
- **Long-horizon:** calibrate T201 in the real Docker sandbox; build T200/202/203/204.
- **safety-under-agency v2:** re-author under audit-report framing (echoing config
  is the ostensible job) — that's what makes claw's real T28 leak, not the naked ask.
- **code-exec:** accept 35B strength or escalate to adversarial-novel (not just
  non-canonical). Lower priority — coding already has the sandbox coding-agentic
  spread.
- **Re-baseline** all standing models once suites are ladder-confirmed; new dated
  dir in `evals/baselines/`, update the comparison table + FOCUS status.

## Trajectory-dataset tie-in

Every deterministic grader here logs per-task verifier outcomes; the long-horizon
sandbox path logs per-step tool dispatches. These are the substrate for the
verified-trajectory dataset (`experiments/agentic-data/RESEARCH.md` play #1) —
eval authoring and dataset construction are the same proposer–solver–verifier loop.
