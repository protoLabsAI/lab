# AgentWorld as the environment-half of self-scaffolding RL

**Status:** proposal (2026-06-28). Document-only — no training yet. Grows out of the AgentWorld
fidelity probe ([RESULTS.md](RESULTS.md)) + the Ornith-1.0 release. Our slice if pursued: the
**model foundry** (small Ornith-style agent + the RL/eval loop), not a frontier coding model.

## The observation that started it

Ornith-1.0 (DeepReinforce, MIT) and Qwen-AgentWorld (Qwen, Apache-2.0) are **two fine-tunes of the
same Qwen3.5 base, pointed in opposite directions**, with the *same* size lineup (9B/31B/35B-MoE/
397B-MoE vs 35B-A3B/397B-A17B). Our daily driver `Ornith-1.0-35B` and AgentWorld-35B-A3B are the
same hybrid DeltaNet+attention 256-expert architecture.

- **Ornith is the agent.** Self-scaffolding RL: each step it (1) proposes/refines its own scaffold
  (memory, error-handling, orchestration), then (2) rolls out a solution; reward flows to both,
  high-reward scaffolds are mutated+selected. Async pipeline-RL with staleness-weighted token-GRPO.
- **AgentWorld is the environment.** A world model that, given an action + history, predicts the
  next observation across Terminal/SWE/MCP/Web/OS/Android/Search.

RL needs (policy, environment, reward). Ornith's loop is starving for a fast, controllable
**environment**; AgentWorld is built to be exactly that ("controllable simulation of thousands of
environments for agentic RL… gains surpass real-environment training alone"; "world-model training
as RL warm-up"). The two halves compose into the loop that produces an Ornith.

## The central tension (and the design it forces)

Ornith's capability gains are inseparable from its **three-layer reward-hacking defense**, and
**layer 1 is an immutable environment**: "the environment, the tool surface, and test isolation are
immutable and outside the model's reach." Layers 2–3 (deterministic boundary monitor; frozen-LLM-
judge veto) sit *on top of* a ground-truth verifier.

A world model is the opposite of an immutable boundary — it is learned, fallible, and gameable. Our
fidelity probe quantified the failure modes:

- **AgentWorld reproduces a sandbox's *shape*, not its *state*.** (T103 step 7: asked to simulate
  `SELECT *`, the real output was 50 rows of fixture data — `alice@example.com`, exact timestamps —
  which the model correctly refused to fabricate.) Parse-rate 0.71, but exact/structural match low,
  dominated by unknowable fixture content.

**Implication:** AgentWorld can NEVER be the verifier. An agent optimizing a correctness reward
against it would learn to satisfy the simulator's *guess* of the world — reward-hacking the world
model, exactly the failure Ornith spends three layers preventing. The only configuration where
Ornith's defenses survive:

> **AgentWorld = the imagination / exploration buffer for the scaffold-proposal stage.**
> **The immutable real sandbox = the verifier, and the home of all three anti-hacking layers.**

This is the Dreamer/world-model-RL pattern (imagine cheap rollouts to shape policy; ground the
reward in reality), specialized to Ornith's two-stage structure:

| Ornith stage | What it learns | Can AgentWorld drive it? |
|---|---|---|
| 1 — propose/refine scaffold | interaction *shape*: tool-call workflow, error recovery, decomposition | **Yes** — shape is what AgentWorld simulates faithfully; thousands of cheap rollouts |
| 2 — solution rollout + reward | *correctness* on real state | **No** — needs the immutable sandbox; sim reward is exploitable |

## Why this is interesting for *our* lab

- **On-thesis: small, edge-deployable agents.** Ornith-1.0-9B hits **43.1 Terminal-Bench / 69.4
  SWE-Bench-Verified / 63.1 ClawEval** — edge-sized, matching much larger models. "Can a small
  Ornith-style agent be trained more cheaply by warming up scaffold-learning in a world model?" is
  squarely our "small specialized models" lane.
- **The infra already lines up.** Ornith benchmarks on **ClawEval** (our `claw-eval`) and
  **Terminal-Bench/SWE-Bench** (the families AgentWorld simulates). We have AgentWorld serving, the
  Ornith replicas live, and claw sandboxes for the ground-truth verifier.
- **It's the same reward-trust axis as [`game-rlvr`](../game-rlvr/PROPOSAL.md).** Game engine =
  verifiable-but-narrow reward; AgentWorld = scalable-but-untrustworthy-on-state. mythxengine =
  immutable boundary by construction (the `project_game_rlvr_mythxengine` synthesis). The fidelity
  probe is the instrument that says which stage each substrate can safely drive.

## Hypothesis & first experiment (cheap, decisive)

**H:** Warming up the scaffold-proposal stage in AgentWorld (cheap, high-volume) before/alongside
real-sandbox correctness reward beats real-sandbox-only at fixed compute, for a small agent.

**Minimal test — "does sim warm-up move the scaffold at all":**
1. Policy = our **Ornith-1.0-9B** (already strong; isolates the *scaffold* contribution).
2. Take 1 claw/SWE task category. Run N scaffold-proposal rollouts where AgentWorld answers the
   tool calls (no real Docker). Score whether the *evolved scaffold* (not the answer) transfers:
   does a scaffold matured in-sim raise real-sandbox pass-rate vs the unconditioned scaffold?
3. **Verifier + all rewards stay on the real sandbox.** AgentWorld only shapes the scaffold search.

**Decision gate:** sim-warmed scaffold ↑ real pass-rate → build the dual-env loop. No lift / the
agent games the simulator → **publish the negative** ("world-model envs shape interaction shape but
not correctness — here's the boundary," with the fidelity numbers). On-brand either way.

## Empirical result (2026-06-28) — the scaffold-transfer probe ran

3 struggle-zone tasks × {baseline, cold-placebo, sim} × 5 trials, real Docker sandbox as verifier
(full table in [RESULTS.md](RESULTS.md)). Three clean conclusions:

1. **Sim-practiced scaffold > cold (un-practiced) scaffold on 3/3 tasks** (mean +0.24). Robust.
2. **Neither reliably beats *no* scaffold** — sim ≈ baseline except where baseline is floored.
3. **A cold scaffold can catastrophically self-sabotage** (one task: 0/5); the sim scaffold doesn't.

**AgentWorld's value is a regularizer, not a booster:** practice grounds a self-authored scaffold in
the task's real *process* and failure modes, preventing degenerate scaffolds — but it can't lift a
capable agent above baseline because it supplies correct *process*, not correct *values* (the OOD
probe's "tells the truth about process, lies about values," confirmed from the RL side). So the
dual-env design holds with a **bounded** stage-1 role: sim *stabilises/safens* the scaffold search;
reality must still grade correctness. Not powered for significance (n=3, 5 bimodal trials) but the
sim > cold ordering is consistent.

## Risks / honest caveats

- **Qwen-monoculture elicitation trap (central).** Ornith, AgentWorld, *and* our policy are all
  Qwen3.5. Any "gain" needs a cross-family control + contamination-clean eval before we believe it's
  learning, not prior-elicitation (same guardrail as game-rlvr; "Spurious Rewards" arXiv:2506.10947).
- **Coding-game scope line.** This is agentic-*coding* RL — brushes the parked "out of the coding
  game" decision. Deliberate call required before investing; the foundry/RL-loop framing (not "ship
  a coding model") is what keeps it on-thesis.
- **Self-scaffolding RL is heavy.** Async pipeline-RL + dual-env plumbing is a research program, not
  an afternoon. Start with the scaffold-transfer probe above, which needs no RL loop at all.
- **Simulator drift / exploitation.** Even confined to stage 1, if the scaffold over-fits
  AgentWorld's quirks the transfer fails — which is precisely what the first experiment measures.

## Fit

Small specialized models + verifiable/trustworthy measurement (the eval discipline we just
hardened) + quant/serving for cheap agents. Brand line: *"A world model can teach an agent how to
move, but only reality can tell it whether it was right — here's where the boundary actually is."*

Related: [RESULTS.md](RESULTS.md) (fidelity probe), [`../BACKLOG.md`](../BACKLOG.md) §3 (the four
plays), [`../game-rlvr/PROPOSAL.md`](../game-rlvr/PROPOSAL.md) (reward-trust counterpart),
`project_game_rlvr_mythxengine` (immutable-boundary-by-construction), `project_ornith_daily_driver`.
