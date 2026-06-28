# Game-RLVR — small models × verifiable game rewards (mythxengine)

**Status:** proposal + due-diligence (2026-06-28). Cross-substrate (consumes
[`mythxengine-sdk`](https://github.com/protoLabsAI/mythxengine-sdk) /
[`mythxengine-tcg`](https://github.com/protoLabsAI/mythxengine-tcg)); our slice is the
**model foundry** (small fine-tuned/quantized models + the RL/eval loop).

## Context

mythxengine = deterministic, AI-native multi-agent substrate (sealed **byte-identical replay**,
gRPC agent-as-server, per-player redacted observations). Packs = games (survival MMO, 4X, TCG).
mythxengine-tcg targets LOCM + MTG-class, *"AI as players, AI-generated card sets,"* and
**already ships `neural/deadband_neural/`** — an AlphaZero-style net+MCTS self-play pipeline
(`net.py`/`mcts.py`/`selfplay.py`/`train.py`/`export_onnx.py`). Lineage throughline:
*provably-attributable competition via CI-enforced determinism* — explicitly the cure for the
predecessor `pokemonAgent`'s "untrustworthy yardstick" pain.

The question: where do **our** (LLM) models fit, and what's the real fine-tuning opportunity?

## Due diligence (sourced)

### 1. LLM-as-TCG-player is weak — and the wrong tool for a champion bot
- Strongest LOCM bots are **cheap CPU search** (Coac, depth-3 minimax, ~94% win, no GPU, <100 ms/turn) and **deep RL** (ByteDance ByteRL, CoG-2022 winner). No LLM. [arXiv:2305.11814, arXiv:2303.04096]
- LLMs-as-players across TCGs — MTG (mage-bench), Pokémon (PTCG-Bench, arXiv:2605.29653), Slay-the-Spire (arXiv:2410.02829), UNO (arXiv:2509.09867) — **consistently lose to / barely match** search/rule-based bots, hallucinate rules, cost ~$1+/game, run 100s ms–sec — *outside* LOCM's 100 ms / no-GPU envelope.
- LLM-on-LOCM *specifically* is unexplored (open niche), but the general verdict is settled: **search/RL dominates LLMs at card play.**
- → The repo's `deadband_neural` (net+MCTS) is the right tool for a strong bot. **Don't frame our work as "fine-tune a 9B to win the ladder" — it loses to the cheap neural bot we already have.**

### 2. Where games × LLMs *do* work (evidence-backed)
- Games as a **verifiable-reward RL signal that lifts general reasoning** (transferable), not gameplay: **SPIRAL** (Kuhn-Poker self-play → +8–10% math on Qwen3-4B, arXiv:2506.24119); **SPAG** (adversarial language game → uniform reasoning gains, arXiv:2404.10642).
- **RLVR/GRPO is mature + reproducible** (GRPO arXiv:2402.03300; DeepSeek-R1 arXiv:2501.12948; use the fixes — Dr.GRPO 2503.20783, DAPO 2503.14476, GSPO 2507.18071; drop KL for reasoning RL). Feasible on 2 GPUs at ≤9B; TinyZero-class runs <$30.
- **The deterministic engine makes win/loss a verifiable, reproducible reward** — the clean RL signal text-eval never gives. This is the exact pain we hit all day (LLM-judge silent-0.5, claw ±0.40 per-task variance). **This is the substrate's killer property and the real reason it matters to us.**

### 3. The #1 risk for US: Qwen monoculture → elicitation illusion
- "Spurious Rewards" (arXiv:2506.10947): Qwen-Math gains ~as much from **random** rewards as real ones, and it **doesn't transfer** to Llama/OLMo. Much "RLVR reasoning" is *eliciting latent Qwen priors*, not learning.
- We are **Qwen-everything** (Ornith = Qwen3.5). So any RLVR gain is suspect until a **random-reward control + cross-family + contamination-clean held-out eval**. Cheap, mandatory, highest-value guardrail.
- Verifier-hacking is risk #2 — but games give **exact win/loss** (no LLM judge), sidestepping the silent-0.5 bug class we have on record. Determinism + verifiable reward directly cure the "untrustworthy yardstick."
- Size floor: ~1.5B starts, ~3B robust; <~1B RLVR fails to bootstrap; **base-dependent** (Qwen3 self-improves where Llama-3.2-3B barely does — Gandhi et al. 2503.01307; TinyZero).

### 4. Frameworks (2-GPU usable)
- **OpenPipe ART** — lightest, agent-first, GRPO+Unsloth single-GPU, rollouts hit an external env. Best fit.
- **verifiers** (Prime Intellect) — clean env abstraction over an OpenAI-compatible endpoint (our vLLM :8000); envs double as evals.
- **verl / Agent-Lightning** — heavier (Ray), broadest algos; the parked `agent-lightning` work backs onto verl.
- Box note: route rollouts through vLLM; the sm120/FlashInfer fused-MoE wall → use a **dense** policy (our 9B), not bf16 MoE.

## Strategic verdict

- **Don't** chase "LLM champion player" — neural/MCTS owns that (and the repo already does it).
- **Do** use the substrate as what it uniquely is: a **trustworthy, verifiable-reward RL + eval gym**. Two evidence-aligned LLM plays for our lab:
  1. **RLVR reasoning-gym** — small Qwen + GRPO on a verifiable game reward → measure game skill *and* reasoning transfer, with anti-elicitation controls. (Cheap; SPIRAL-backed; tests the core hypothesis.)
  2. **AI worldgen / card-set generator** — LLM generates content → engine self-plays (`deadband_neural` bots) → balance metrics → reward the generator. LLMs' actual strength (generation), closing the "AI-generated worlds" loop with a *verifiable* critic. Higher value, more infra.

## First experiment (recommended): RLVR reasoning-transfer probe

Cheapest test of the core hypothesis, de-risked away from self-play.
- **Goal:** does a small Qwen + GRPO on a deterministic game's win/loss reward (a) get better at the game vs a *fixed* opponent, and (b) transfer to held-out reasoning — *net of elicitation*?
- **Model:** dense Qwen — Ornith-9B, plus 4B/2B for the size-floor question.
- **Env:** `toy_duel` or LOCM via the gRPC `Agent` SDK; opponent = fixed (a `deadband` bot / scripted); reward = win/loss (verifiable, deterministic, byte-replayable).
- **Trainer:** ART or verifiers; GRPO with Dr.GRPO + DAPO fixes; no KL; log entropy every step.
- **Controls (non-negotiable):** random-reward run + a non-Qwen base + a contamination-clean reasoning eval. Without these the result is meaningless.
- **Cost:** a few GPU-hours / <$30-scale.
- **Decision gate:** real (game ↑ *and* reasoning ↑ *and* survives controls) → scale + the worldgen loop. Elicitation-only → **publish the negative** ("games-as-RLVR on Qwen = prior-elicitation, not learning") — itself a valuable, on-brand finding.

## Risks / honest caveats

- **Elicitation illusion (Qwen)** — central scientific threat; controls mandatory.
- **Latency/cost vs the 100 ms/no-GPU game envelope** — fine for research/training, NOT a live cheap ladder (neural bot owns that).
- **Self-play collapse** — start with a *fixed opponent* (RLVR), not self-play; add self-play only if it's the research question.
- **Cross-substrate scope** — this is a program, not an afternoon; our slice is the foundry + RL/eval loop.

## Fit

On-thesis: small specialized models, verifiable/trustworthy measurement (the eval discipline we
just hardened), quant/serving for cheap agents. Cures the recurring reward-noise pain with a
verifiable substrate. Brand line: *"Games give the clean reward signal text-eval can't — here's
what a small model actually learns, and the control that separates learning from prior-elicitation."*

Related: `experiments/eagle3/`, `experiments/mtp/`, `experiments/BACKLOG.md` (#3 AgentWorld),
`project_3way_baseline_t28_safety` (the eval-noise this escapes), parked `agent-lightning`.
