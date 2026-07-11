# agentic-data — landscape study + dataset-contribution proposal (2026-07-02)

**Status:** research report (deep-research harness, 105 agents, 23 primary sources, 24/25 claims
survived 3-vote adversarial verification). Anchor: **Agents-A1** (arXiv 2606.30616, InternScience,
submitted 2026-06-29 — the model we challenger-evaled 2026-06-30, `evals/baselines/2026-06-30-agents-a1/`).
Companion to `experiments/game-rlvr/PROPOSAL.md` and `experiments/agentworld/`.

## The anchor paper, compressed

"Scaling the Horizon, Not the Parameters": a 35B-A3B MoE (same base arch as Ornith) claiming
trillion-param agentic parity (vs GPT-5.5 / DeepSeek-V4-pro / Kimi-K2.6) via:

1. **Knowledge-Action Infrastructure** — per-domain typed 4-tuple (corpus, actions, observations,
   verifiers); a Knowledge-Action Graph expanded by proposer–solver–verifier self-play. Trajectories
   admitted only if verifiable against explicit checks, evidence actually consulted, no shortcut
   solutions; failures kept as negatives. Humans only final-filter.
2. **~100K trajectories averaging 45K tokens** (search 44K, coding 48K, science 37K) → full-domain SFT.
3. **Six domain teachers** (SFT or GRPO per domain — search teacher = GRPO on only ~2K questions) →
   **domain-routed on-policy distillation** with hard routing + Salient Vocabulary Alignment
   (truncated reverse-KL on the teacher's top-k support). Their own tables: the OPD student does
   NOT always beat its domain teacher.

**What they release: weights + eval code. What they keep: the trajectory corpus, all six teachers,
the harnesses.** The field's key artifact is closed (verified against their GitHub + all 22 HF
datasets in the InternScience org — only benchmarks, no training data).

Our independent read (challenger run 2026-06-30): ties Ornith on claw (0.734 vs 0.741), FC 91%,
code-exec 0.97, 208.7 tok/s — all that infrastructure bought parity-with-Ornith on *our* horizon
lengths; their wins are claimed on long-horizon benches (GAIA/BrowseComp/MLE-class) that claw
doesn't measure. Treat "trillion-param parity" as promotional; the *data recipe* is the real content.

## Verified landscape findings (what the field converged on)

| # | Finding | Source | Why it matters to us |
|---|---|---|---|
| 1 | Frontier agentic data is **constructed, not collected** — synthesized trajectories with built-in verifier outcomes | Agents-A1 (2606.30616) | Our verifiable-reward substrates ARE data factories |
| 2 | **~1K prompts with 4:3:3 easy:med:hard is the RL sweet spot**; 2K prompts *degrades* OOD (35.0→32.2) | 2603.21972 (2-1 vote; single 3B/TravelPlanner study — narrow) | Small-lab data scale is *sufficient*, not a handicap |
| 3 | **Adversarial env synthesis works at scale**: CUA-Gym generator/discriminator co-writes env + deterministic reward (reward(golden)=1.0, reward(initial)=0.0 under execution) → 32,112 verified RLVR tuples, 110 envs | 2605.25624 + xlang-ai repo | Closest published template for a game-RLVR gym release |
| 4 | **Mock-env training transfers to real benchmarks**: CUA-Gym A17B → 72.6% OSWorld-Verified; SynthAgent-8B (GRPO on 15,096 fully LLM-simulated tasks, local open models) beats Qwen3-32B (42.9 vs 36.0, TAU-2+BFCL) | 2605.25624, 2601.22511 | Existence proof the whole pipeline fits on 2×96GB |
| 5 | Every mock-env paper **asserts fidelity, none measures it** ("high-fidelity" is self-described everywhere) | 2605.25624, 2601.22511, 2603.21972 | AgentWorld's differential-fidelity study is an unoccupied niche |
| 6 | The only quantified sim2real gap is **user** simulation (best USI 76.0 vs human 92.9, 31 sims tested) — **environment** fidelity is unmeasured | 2603.11245 | Confirms #5 from the skeptical angle |
| 7 | Reward-hacking eval is moving to **hack-verifiable-by-design environments** (embedded exploits, deterministic detection) vs post-hoc trajectory inspection | Hack-Verifiable TextArena (2606.26300-cluster) | Direct prior art for game-rlvr's reward-trust framing |
| 8 | **SWE trajectories are saturated**: microsoft/Orchard = 107,185 trajectories w/ execution-verified resolve labels across 2,788 repos; SWE-ZERO-12M = 12.3M rollouts / 112B tokens; + nebius/SWE-agent-trajectories, AgentTrove | HF datasets | Do NOT build SWE trajectory data — the gap is elsewhere |

Refuted in verification (dropped): "reward/algorithm choice is scale-dependent (small→curriculum+ARPO/DAPO,
7B→dense+GRPO)" — 1-2 vote, don't cite.

**Coverage gaps of this study (honest):** Kimi-K2.6 / DeepSeek-V4 data-recipe internals, MLE-Dojo,
τ²-Bench lineage, and a complete open-dataset census did not survive verification — sanity-check any
committed play against HF dataset search first. All headline numbers are self-reported preprints
days-to-weeks old.

## Where the open gap actually is

Saturated: SWE/coding trajectories (Orchard, SWE-ZERO), generic tool-call SFT (ToolBench-class),
benchmark suites. Closed-but-proven: long-horizon multi-domain verified trajectories (Agents-A1,
and likely Kimi/DeepSeek equivalents). **Open and matched to us:**

1. Verified-reward trajectories in **non-SWE, deterministically-verifiable domains** (game engines,
   protocol/parsing tasks, our code-exec-graded suite) — nobody publishes these with per-step
   verifier outcomes + negatives + replay determinism.
2. **Environment-fidelity measurement** — the assumption load-bearing for findings 3–4, measured by no one.
3. **Quantization × agentic capability** — zero published work on whether FP8/INT4/IQ2 degrades
   long-horizon agency differently than single-turn scores (does error compound over 45K tokens?).
4. **≤9B distillation traces** — OPD data from strong agentic teachers, at 2-GPU scale.

## Ranked plays (synthesis — the recommendation layer, not externally verified)

| rank | play | builds on | cost | why us |
|---|---|---|---|---|
| **1** | **`protoLabs-agentic-verified-v0`**: 1K–10K trajectories from game-rlvr (byte-replayable win/loss) + claw sandbox tasks via code-exec grader, published with per-step verifier outcomes, failed-attempt negatives, difficulty mix per finding #2, proposer–solver–verifier construction per Agents-A1/CUA-Gym | game-rlvr PROPOSAL, code_exec.py, Ornith replicas (~6500 tok/s aggregate gen+verify) | med | Fills the exact artifact Agents-A1 closed, in domains SWE-saturation doesn't cover; our reward-trust discipline (random-reward controls, no LLM-judge in the reward path) is the differentiator |
| **2** | **Environment-fidelity benchmark** (AgentWorld scaled): sim-vs-real divergence per domain, claw ground truth, "derivable vs hidden" boundary + LLM-judge fidelity scoring; publish tasks + probe harness | agentworld/ (probe built, findings in RESULTS.md), BACKLOG §3 play #3 | low-med | Every mock-env paper needs this number and none has it; experiment already in flight |
| **3** | **Quant × agentic study**: bf16 vs FP8 vs INT4 vs IQ2 on claw + long-horizon tasks, error-compounding curve vs trajectory length | quantize/ pipeline, parity-verification method, low-bit-35B finding | low | Zero competition; composes our two proven strengths; pure eval-compute |
| **4** | **OPD-at-2-GPU distillation traces**: Ornith-35B/Agents-A1 teachers → Ornith-9B student, publish traces + SVA-style recipe notes | ornith-9b line, LLaMA-Factory/TRL | high | Highest compute; gated on #1 for data; the OPD-student-vs-teacher tradeoff (paper's own admission) is the research question |

**Sequencing:** #3 and #2 are cheap and independently publishable — start now, they de-risk nothing
and feed the blog pipeline. #1 is the flagship dataset release and consumes the game-rlvr experiment
as its substrate (the RLVR probe's rollouts *are* the dataset's first shard — instrument from day one:
log per-step verifier outcomes, keep negatives, record replay seeds). #4 waits on #1.

**Controls carried over from game-rlvr (non-negotiable for #1):** random-reward control, non-Qwen
base check, contamination-clean held-out eval — the Spurious-Rewards/elicitation guard applies to
the *dataset's advertised value*, not just our own training runs.

## Sources (primary)

Agents-A1: arxiv.org/abs/2606.30616 · huggingface.co/InternScience/Agents-A1 · github.com/InternScience/Agents-A1
Data construction: 2603.21972 (RL recipe/TravelPlanner) · 2605.25624 + github.com/xlang-ai/CUA-Gym · 2601.22511 + github.com/haruhi-sudo/SYNTHAGENT
Reward-trust/skeptical: 2603.11245 (sim2real user-sim) · 2606.26300 · 2604.15149 · 2605.07247
Open datasets: huggingface.co/datasets/microsoft/Orchard · nebius/SWE-agent-trajectories · AlienKevin/SWE-ZERO-12M-trajectories · 2505.19433
Full verification trail: deep-research run wf_443c3ed2-56e (105 agents, 24/25 confirmed 3-0 or 2-1).
