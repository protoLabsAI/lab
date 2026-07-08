# MythXEngine × distill/RL bridge — synthesis + first-integration sketch (2026-07-07)

**Where all this is headed.** τ-bench was the borrowed sandbox to *prove the loop* (Ornith rollouts →
deterministic verified reward → reward-filter → distill → gate). **MythXEngine (`~/dev/mythxengine-sdk`)
is the same loop on a substrate we OWN** — stronger reward-trust, long-horizon + multi-agent by
construction, an infinite diverse-world factory instead of ~3,300 fixed τ-tasks. This is the answer to
the τ-bench scaling ceiling (see [[project_distill_base_decision]] scaling conclusion).

## What MythXEngine is
A **deterministic Rust ECS substrate for multi-agent autonomous-agent research in persistent MMO-shaped
worlds.** Genre-agnostic; environments are **packs** (data + components + systems + scenarios). Turn-based
with continuous ambient world systems (bounded LLM-paced decision windows). Load-bearing promise: for
`(engine_version, pack_versions, scenario_version, seed, locked_plan_batches)` → **bit-identical
state/events/diffs across runs, with an independent replay verifier.**

**The RLVR pivot is already underway** — `docs/plan/agent-zoo.md` (dated 2026-07-07) frames the engine as
an **"RLVR environment factory,"** and **Ornith-35B has already been run through it**: reward-labeled
trajectories on disk at `/mnt/data/datasets/agent-zoo/{courier,arena}-ornith35b-v0.jsonl` (186 rows).

## What's DONE (the hard parts — credible)
- **Judge-free reward trust (the crown jewel).** `substrate/src/replay.rs` + `session_verifier.rs`
  replay the *recorded actions* through the pure resolver and assert byte-identical events — they do NOT
  re-invoke agents. LLM nondeterminism is irrelevant; the reward channel **cannot be gamed by a
  nondeterministic policy, and every reward is independently re-derivable from the recording.** This is
  *stronger* than τ-bench's DB-check — the immutable-boundary property the game-RLVR synthesis wanted
  ([[project_game_rlvr_mythxengine]]).
- **LLM plugs in today.** `examples/llm_band.py` (PR #317): one model commands a team via OpenAI
  function-call tool schemas + ReAct loop, talks to local vLLM, writes every window to `traces.jsonl` in
  student-harness shape. gRPC bidi transport (`proto/mythx.proto`) + Python `Agent` SDK
  (`python/mythx_sdk/.../agent.py`).
- **Trace→reward-labeled-SFT pipeline exists.** `examples/zoo_traces.py` fuses traces + recorder +
  leaderboard → `{messages, tools, reward:{alive,banked,rank,baseline_ratio}, meta}` rows,
  gauntlet-normalized.
- **Proven RL muscle (game nets).** `neural/` = AlphaZero (Rust MCTS self-play + PyTorch net + ONNX
  serve). `neural_frontier/` = PPO/league, with the **filesystem-boundary self-play↔trainer pattern on
  Blackwell** (trainer writes safetensors, Rust self-play writes games.jsonl) — the reusable RL-rollout
  shape.

## What's MISSING — and it's exactly what WE (lab) already built
1. **No LLM training loop.** BC (R2) + GRPO/RLVR (R3) over the traces don't exist in the SDK repo —
   "presumably in the lab repo." **That's the pipeline we built this session** (reward-filter →
   masked-SFT → NVFP4 → gate). It ports directly. We bring training; MythXEngine brings the env factory.
2. **Reward is coarse/terminal-only.** Needs dense verifiable **subgoal events** — and the attribution
   infra already exists (`worldplay/src/leaderboard.rs` does `deposited→banked`). Lowest-friction,
   highest-impact change.
3. **The one built world (COURIER) gave a NEGATIVE result** — the load-bearing lesson: greedy baseline
   **36** vs commander-harness **21** vs LLM-flat **9**. Greedy-beatable = **"shallow by construction."**
   Effort spent on a task with no learnable headroom.

## Maturity (honest)
**"R1 of R4."** Substrate mature + battle-tested; RLVR-gym layer days old, one shallow world, transfer
hypothesis (does playing worlds → real agentic skill) untested. `prd.md` still literally says "not a
gym-style environment library" — the RL framing is a *very recent* reframe.

## Highest-leverage builds (ranked)
1. **Dense subgoal-reward events** in packs (infra exists — near-free). Terminal 0/1 → judge-free dense.
2. **Wire our distill/RL pipeline to the rollout farm** — parallel `mythx_host serve` → traces +
   verifier reward → our GRPO trainer. Reuse the `neural_frontier` filesystem-boundary pattern (proven).
3. **Greedy/frontier/ORACLE headroom triplet per world** — gate GPU spend on a *proven learnable gap*
   before building/training. COURIER is the cautionary tale.
4. **Schematize the harness view** (compact obs + tool surface) into the pack — trace stability = data
   integrity for multi-week runs (currently lives in Python `_compact_obs`, can drift).
5. **2–3 genuinely deep worlds** with proven headroom — FOREMAN (gather→refine→deliver→build dependency
   chain), SLEUTH (certified puzzle). Current worlds are ~40-window forage loops (~90% "keep walking").

## First-integration sketch (the bridge from today → here)
The cheapest real step that proves the MythXEngine→distill path end-to-end, reusing what we have:

1. **Adapter: agent-zoo trace format → our canonical `Trajectory`.** `zoo_traces.py` already emits
   `{messages, tools, reward{...}}`; write a `dataset/adapters.py` entry (`_agent_zoo`) mapping it to our
   schema. Reward-filter on the verifiable fields (rank/baseline_ratio ≥ threshold) exactly like
   `filter_tau2.py`. → drops agent-zoo trajectories straight into `build.py`.
2. **Headroom gate FIRST (learn from COURIER).** Before generating a big corpus, run the greedy gauntlet
   + an Ornith-frontier pass + (if buildable) an oracle bound per candidate world. Only worlds with a
   wide greedy→frontier gap are worth Ornith rollouts. Publish the triplet as the world's spec.
3. **Generate + distill on a PROVEN-HEADROOM world** (not COURIER v0). Ornith rollouts via `llm_band.py`
   → reward-filter → our masked-SFT (or white-box KL, below) → NVFP4 → gate on *held-out world seeds*
   (byte-replay = contamination-clean by construction).
4. **Then RL (R3):** parallel `serve` rollout farm → verifier reward → GRPO (the `rlvr-poc` plan), which
   is the lever that can exceed the teacher.

**Contamination note:** MythXEngine's determinism makes clean held-out trivial — eval on unseen `seed`s
of the same world; the verifier guarantees the eval episodes are independent and re-derivable.

## Why this resolves the scaling-cap finding
The ×3-confirmed τ-bench lift was small (+0.028) with diminishing returns — SFT-on-fixed-tasks flattens.
The three levers out: **white-box KL** (more signal, same data — [[project_distill_base_decision]]), **RL**
(exceed teacher), and **our own verified-reward env factory** (past the diversity ceiling). MythXEngine is
the third and biggest — and it's ours, with better reward-trust than the borrowed sandbox.

## Pointers
- SDK: `~/dev/mythxengine-sdk` — `docs/plan/agent-zoo.md` (vision + honest self-critique),
  `examples/llm_band.py` + `zoo_traces.py` (harness + trace→SFT), `substrate/src/{replay,session_verifier}.rs`
  (reward-trust), `worldplay/src/leaderboard.rs` + `substrate/src/scorecard.rs` (verifiable reward),
  `neural_frontier/README.md` (reusable RL pattern).
- Our side: `experiments/agentic-data/` (distill pipeline — the R2 half), `experiments/rlvr-poc/` (R3
  plan), `experiments/game-rlvr/` (the gym proposal this realizes).
