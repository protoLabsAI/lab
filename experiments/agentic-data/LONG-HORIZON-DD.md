# Long-horizon verified-reward — due diligence (2026-07-07)

**Question that triggered this:** the current pipeline is rejection-sampling SFT (Ornith-35B →
reward=1.0 filter → SFT Qwen-2B). That's medium-horizon (τ-bench: a few tool calls, terminal
DB-check). Where do we go for **long-horizon** tasks where "reward = success" is too sparse and too
coarse — and is that a curriculum? Three parallel research streams (env catalog, curriculum/iterated
methods, process-reward decomposition) + our own prior art. Sources cited inline.

## TL;DR — three moves, in dependency order

1. **Densify the reward we already have** (cheapest, highest leverage, no new data). τ² already
   computes N deterministic checks per task but **AND-composes them as a product**, collapsing to
   0/1. Swap product → fraction-of-checks + potential-based shaping. This is a code change, not a
   research project.
2. **Recover the discarded rollouts** via deterministic/teacher **hindsight relabeling** — the #1-EV
   fix for filter-starvation, and it gets *stronger* as horizon grows (more failures to reclaim).
3. **Only then reach for a new long-horizon env** — and the pick is **AppWorld** (Apache-2.0,
   longest deterministic horizon, offline, low friction), not another τ-domain.

Curriculum is real but **shallow** (plateaus in 2–3 iterations across every source) — treat it as a
modest multiplier on 1+2, not a self-improvement engine. **Composition > provenance > verification >
scale** (our pilot) still governs; nothing here overturns it.

---

## 1. Densify the reward (process/checkpoint decomposition)

**The finding that matters, read straight from our τ² code** (`src/tau2/evaluator/`): the final
reward is a **product** over the deterministic components —

```
reward = db_check × action_check × env_assertion × communicate     # each 0.0 or 1.0
```

and even *within* env-assertions it multiplies per-assertion booleans. So we already author a
decomposed, machine-checkable signal per task — and then throw the partial-credit away by ANDing it.
That AND *is* the terminal-sparse credit-assignment problem, just spelled differently.

**Densification ladder (all stay deterministic, no LLM in the reward path):**

- **(a) Fraction-of-checks** — weighted mean over the booleans we already emit (`reward_breakdown`
  already carries them, currently unused). "3 of 5 checks → 0.6." Cheapest; do first.
- **(b) Per-entity DB predicates** instead of one whole-DB hash — "order X == shipped", "balance ==
  Y" — so touching 5 rows and getting 4 right ≠ 0.
- **(c) Monotone-prefix / milestone chain** for genuine sequential long-horizon — reward the longest
  satisfied prefix. Monotone-prefix **resists gaming** (can't harvest checkpoint 4 unless 1–3's
  state is genuinely present) better than a bag-of-independent-checkpoints sum.
- **(d) Checkpoints harvested for FREE from the gold replay.** We *already* run the gold environment
  over `evaluation_criteria.actions` to compute the target hash — snapshot per-action intermediate
  state from that same replay → an ordered checkpoint chain at **zero authoring cost**. This is the
  highest-ROI densification source; it's sitting in the harness unused.

**Keep it optimality-safe:** add the dense signal as **potential-based reward shaping** (PBRS, Ng
1999): `F = γΦ(s′) − Φ(s)` with `Φ(s) = #checkpoints satisfied`. Provably preserves the optimal
policy of the true terminal-reward MDP, so we get dense credit assignment *without* optimizing a
proxy. (Confirmed in the multi-turn-RL practitioner guide, arXiv 2510.01132. Caveat: the invariance
holds for state-potentials — `#checkpoints reached` qualifies; don't make Φ depend on action/history.)

**Anti-gaming / anti-over-constraint rules:**
- Keep the **true terminal reward dominant**; validate a full-success trajectory strictly beats every
  partial one.
- **Prefer path-agnostic end-state predicates (ENV basis) over action-sequence matching.** Action
  matching punishes valid alternative solutions and kills exploration. τ²'s DB/env-assertion checks
  are already path-agnostic; the *action* checks are prescriptive — lean on the former for RL.
- **Down-weight read-only actions** (cheap to spam) using the `tool_type` (read/write/think) tag τ²
  already records; weight irreversible writes highest.
- If we ever auto-generate checks, **gate each through an execution test** (pass-on-gold,
  fail-on-known-bad) before it enters the reward — auto-checks tend to be redundant/over-prescriptive
  (Agentic-Rubrics, 2601.04171).

**Refs:** Verifiable Process Rewards 2605.10325 · practitioner guide 2510.01132 · Agent-RLVR
2506.11425 (keeps reward strictly binary-verifiable, uses LLM only for *training-time guidance*, never
scoring — the discipline to copy) · DeepSWE (test-execution RLVR) · PBRS (Ng et al. 1999).

## 2. Recover discarded rollouts — hindsight relabeling (the starvation fix)

On long tasks the filter bites hard: terminal success might be ~15%, so RS-SFT starves. The dominant,
best-evidenced fix is **hindsight relabeling** — turn the failed 60–85% into training signal:

- **STaR rationalization** — re-run the *teacher* on a failed task **with the gold answer/goal
  revealed**, keep the trajectory it now produces, SFT the student on it. Manufactures a correct
  trajectory for items the student can't yet solve, so the hard tail doesn't silently drop.
- **AgentHER goal-reverse-engineering** (2603.21357) — for a failed trajectory, have the teacher infer
  a goal it *did* satisfy (multi-judge verified, ~97.7% precision) → emit SFT/DPO data. **+7–12pp over
  success-only SFT, 2× data efficiency, and smaller models benefit most.** Directly on-point for a 2B
  student.

**Why this is the #1 move for long-horizon specifically:** its value *grows* with horizon — the
longer the task, the more failures, the more there is to reclaim. It's pure SFT (no RL infra). **The
one hard requirement: relabel verification quality.** Reverse-engineered goals can be wrong;
require multi-judge agreement or a deterministic re-check, or you inject 6% label noise. And mix in
un-rationalized successes so the student doesn't learn to lean on hints it won't have at test time.

**Note vs our reward-trust doctrine:** relabeling uses an LLM to *construct training data*, not to
*score reward* — same line Agent-RLVR draws. That's inside our rules (we distrust judges in the
**reward path**; data construction with a verify gate is fine, and is exactly what τ-bench rollout
generation already is).

## 3. Curriculum & iteration — real but shallow

- **Difficulty mix beats volume.** The RL sweet spot is **~1K prompts at ~4:3:3 easy:med:hard**; past
  ~1K, in-domain keeps rising but **OOD generalization degrades** (2603.21972 — and our own
  agentic-data RESEARCH #2 independently logged the same finding). More filtered data is *not*
  strictly better. This is the small-lab-is-sufficient result restated.
- **Best difficulty signal = rollout pass-rate binning** (free from rollouts we already generate);
  **horizon/step-count** is the right proxy specifically for long-horizon; LLM-judged difficulty is a
  near-free alternative. Adaptive bandit ordering (Self-Evolving Curriculum) beats static binning only
  modestly — skip for a first prototype. Heed "Difficulty Is Not Enough" (AAAI): weight by *utility*
  (does the sample move the student), not difficulty alone.
- **Iterated RS produces an emergent curriculum, but it plateaus fast.** ReST-EM diminishes after
  1–2 iterations; AgentHER tapers +1.6pp→+0.5pp by round 3; flywheel methods plateau in "a few
  cycles." **Budget 2–3 rounds.** Two guards against collapse: (1) **restart each SFT from base**, not
  the previous checkpoint (ReST-EM's explicit anti-overfit guard); (2) keep **injecting fresh/harder
  items** so the loop isn't refiltering the same easy trajectories (distribution narrowing).
- **Dense→sparse reward schedule** — the 2603.21972 recipe (small students!) anneals partial-credit →
  terminal-only over epochs and takes a 1.5B from **6.9% → 34.9%**. The SFT-only slice lands lower
  (their headline needs the RL stage), but the data-curation half is most of the small-model gain.

**Failure modes to watch:** fast plateau/overfit on a small static problem set; distribution
narrowing/mode collapse from refiltering similar successes; easy-task bias if the difficulty threshold
never advances (this *re-creates* starvation); OOD drop past the ~1K sweet spot; relabel noise.

## 4. Env catalog — where to get long-horizon deterministic reward

Ranked by (long-horizon × deterministic reward × low friction × permissive license). Full table in
the research appendix; headlines:

    rank  env            horizon      reward                       license    friction   why
    1     AppWorld       L (~42, ≤244 PROG state-unit-tests +       Apache-2.0 LIGHT      longest det. horizon; has a
                         API calls)   collateral-damage checks,     (offline)             TRAIN split; fills the exact
                                      decomposes TGC/SGC                                  non-SWE gap; local vLLM
    2     τ²/τ³-bench    M-L (100-200 PROG DB-hash + comm-string;   MIT        MED        what we already run; dual-
                         dual-control)NL_ASSERTION judge OFF by     (on disk)             control telecom = longer
                                      default for our domains                             horizon; user-sim = friction
    3     AgentGym       S-M (≤30)    PROG all 14 envs, no judge;   MIT        MED        best DISTRIBUTION diversity;
                                      ships 6.1k-14.5k SFT traj +                         Verl RL + ScalingInter
                                      Verl RL framework                                   horizon curriculum built in
    4     TravelPlanner  M (30-step)  PROG per-constraint pass      MIT        LIGHT      best DENSE reward gradient
                                      rates (dense sub-scores)      (offline)             already decomposed; brutal
                                                                                          headroom (GPT-4 ~0.6%)
    5     BALROG         L-XL (1e2-   PROG engine-progress 0-100,   MIT        LOW        true long-horizon + native
                         1e5 steps)   fine-grained per-game                               vLLM; but GAMES not tool-use
                                                                                          (transfer risk)

**Bonus clean-reward templates:** WorkBench (exact-DB-state, MIT, ~saturated → good *template*),
DABStep (numeric-tolerance code analysis, CC-BY, very hard), BFCL (AST+exec, short-horizon),
Spider 2.0 (SQL execution-accuracy).

**Disqualified — LLM-judged reward (launder hallucinations, against doctrine):** Online-Mind2Web
(WebJudge), BrowseComp, ToolBench, StableToolBench, ToolEmu, AgentBench-LTP. **Filter-before-use:**
WebArena/VWA (mostly programmatic but a GPT-4 `fuzzy_match`/VLM `eval_vqa` subset — partition out),
Mind2Web-offline (deterministic but scores against one recorded trace → brittle to valid alt paths),
GAIA (deterministic final-answer string only — ignores the path, so a hallucinated answer via a wrong
path still scores 1; also needs live web).

## How this maps to what we already own (don't rebuild)

This DD lands on top of four staged internal experiments — the plan is to *converge* them, not start
fresh:

- **`experiments/game-rlvr/`** (byte-replayable verifiable-reward gym) + **`agentic-data` play #1**
  (`protoLabs-agentic-verified-v0`: publish trajectories with per-step verifier outcomes + failed
  negatives + replay seeds, proposer→solver→verifier construction per Agents-A1/CUA-Gym). The
  densified τ² reward (§1) + hindsight negatives (§2) are exactly the per-step-outcome + kept-negatives
  this dataset advertises. **Instrument the τ² generation for this NOW** (log per-check breakdown, keep
  reward<1.0 rollouts) — the telecom run in flight *is* the first shard.
- **`experiments/rlvr-poc/`** (GRPO on our graders, reward-hardening) — the natural consumer of the
  densified reward once we go past SFT. Its Gate-1 (immutable tests, zero-and-exclude monitor) is the
  anti-gaming layer §1 assumes.
- **`experiments/agentic-coding-rl/`** — sharp correction already banked: **for RL reward the field
  uses sparse binary all-tests-pass, NOT partial credit** (DeepSWE/Kimi-Dev). Reconcile with §1: use
  **partial credit as PBRS *shaping* over an unchanged sparse terminal objective**, not as the
  objective itself. That's the reconciliation — shaping ≠ objective.
- **CUA-Gym** (in our RESEARCH.md): generator/discriminator co-writes env + deterministic reward
  validated by `reward(golden)=1.0, reward(initial)=0.0`. That's the template for §1(d)
  auto-checkpoint construction *with* a gaming guard, and for authoring new domains cheaply.

## Recommended next actions (cheap→expensive)

1. **[cheap, now] Instrument the current τ² generation** to dump `reward_breakdown` per sim and keep
   the reward<1.0 rollouts (don't discard). Zero cost, unblocks §1/§2 and feeds dataset play #1.
2. **[cheap] Prototype the densified reward** — a `reward_dense()` that reads the existing per-check
   booleans and returns fraction/monotone-prefix instead of the product. Validate: full-success still
   strictly dominates; a gold-trajectory replay scores 1.0, a known-bad scores <1.0. Pure offline.
3. **[med] Hindsight-relabel a batch** of the kept failures (teacher rationalization + verify gate);
   measure SFT lift vs success-only at matched N. This is the direct long-horizon starvation test.
4. **[med] Stand up AppWorld** (Apache-2.0, offline, has a train split) as the first genuinely
   long-horizon deterministic env → Ornith rollouts → densified reward → the same distill loop.
5. **[later] Wrap 2–4 in a 2–3-round ReST-EM loop** (restart-from-base, widening difficulty pool)
   only after single-shot lift is confirmed — expect a modest multiplier, not magic.

## Executed (2026-07-07) — actions #1 + #2 done

**#1 (keep negatives + breakdown) — already free.** τ² natively writes *all* sims to `results.json`
with the full per-check breakdown (`env_assertions[].met`, `action_checks[].action_match` +
`tool_type`, `db_check`, `communicate_checks`, `reward_basis`). Nothing was lost; no instrumentation
needed. **Clarification it forced:** "reward<1.0" is NOT the failure count — of 1645 telecom sims,
164 are **ungradable** (empty `reward_basis`, no criteria) and only **~50 are genuine gradable
failures** (~3% — telecom really is that easy for Ornith). Don't count empty-basis tasks as negatives.

**#2 (`reward_dense()`) — built + validated on real telecom data** (`reward_dense.py`). Recomputes
each basis component as a *fraction* (or monotone-prefix) of the deterministic checks τ² already
emits, respects `reward_basis`, excludes NL_ASSERTION, down-weights read-only actions by `tool_type`.
Self-test on 1645 sims:
- **INVARIANT holds: 0 breaches** — every official-success sim scores dense = 1.0 (full success
  strictly dominates → safe as PBRS shaping over the unchanged 0/1 objective).
- **32% of genuine failures recovered a >0 gradient** (16/50), all in the (.25,.5] band (they
  satisfied 1 of 2–3 env-assertions); the other 34 satisfied *none* → correctly still 0.
- **The modest recovery is the domain, not the method:** telecom is nearly all-or-nothing on 1–2
  env-assertions (few checks = little partial surface). The payoff scales with checks/task — airline
  (~48% fail) and especially a long-horizon env like AppWorld (avg 8 checks/task) are where the dense
  gradient actually lives. Telecom validated *correctness*; it can't show much *recovery*.

Remaining: #3 hindsight-relabel a batch of the 50 (+ airline's larger failure pool) and measure SFT
lift vs success-only; #4 stand up AppWorld. Both gated on GPU (telecom scale-run has the cards).

**Honesty flags:** curriculum/iteration gains are shallow and plateau in 2–3 rounds everywhere;
bandit-curriculum barely beats plain pass-rate binning; the one best-matched small-student result
(6.9→34.9%) needs an RL stage we'd have to reproduce. Several env/method citations are 2025–26
preprints (τ³ numbers are Sierra-reported/future-dated — but its reward *mechanism* is verified from
the repo on disk). The τ² code findings are read directly and exact.
