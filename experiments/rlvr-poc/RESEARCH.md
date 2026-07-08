# RLVR-PoC — a verifiable-reward RL loop on our own graders

**Thesis:** we've spent our cycles making models *smaller/faster* (quant + serving). RLVR is the move *up the stack* — making them *better* — and it closes our loop: our eval graders stop being a scoreboard and **become the training signal**. It unifies three parked threads: `game-rlvr` (verifiable-reward gym), `agent-lightning` (GRPO plan), `agentic-data` (verified trajectories). We are **not starting from zero** — we already own the reward half.

## What Ornith actually does (documented; not reproducible)

DeepReinforce (Jiwei Li's lab; prior: CUDA-L1). Ornith = agentic-coding family, Qwen3.5/Gemma4 bases, MIT.
- **Self-scaffolding RL:** the model generates *both* the solution rollout *and* its own task-specific harness/scaffold; reward flows to **both** → scaffold + policy co-evolve.
- **Token-level GRPO**, async with **staleness-weighted** off-policy tokens (age down-weight + discard threshold) for long-horizon stability.
- **Execution-grounded reward + 3-layer anti-reward-hack:** (1) fixed trust boundary isolating the exec env, (2) deterministic monitor that zeroes reward *and excludes the trajectory from advantage* on gaming (reads withheld paths / edits verifier), (3) frozen LLM judge overriding on detected gaming. **← Layer 1 == our `mythxengine` immutable-boundary note.**
- **NOT released:** paper, training code (GitHub is inference-only), data, compute, hyperparams. Recipe *class* is legible; exact repro is off the table.

## The PoC (GO — homelab-sized, overnight)

Framed as a **serving + reward systems experiment**, NOT a SOTA chase. The publishable result is the systems story (NVFP4-rollout throughput, grader-as-reward + hardening, honest saturation/reward-hacking findings) — "patterns to study and steal," not "we beat Qwen."

| piece | choice |
|---|---|
| framework | **TRL `GRPOTrainer`** (rank 1 — lowest friction, we know it, best-documented) |
| policy | Qwen3.5-**2B** or **4B**, **QLoRA** |
| reward | **`evals/graders/code_exec.py`** (pass-rate + partial credit, clamp [-1,1]) + format/coherence penalty |
| rollout | **colocated vLLM, sleep-mode** — one card generates (NVFP4 = rollout-speed lever), one trains; `num_generations≥8` |
| data | Phase-3 `code-exec-v2` set as seed; expand to a few hundred–few thousand tasks w/ hidden tests |
| compute | 2B/4B QLoRA GRPO, few hundred steps → **overnight** on our rig |

Escalation path (only if PoC shows a real, non-hacked reward curve): **veRL** (mature agentic ecosystem — VerlTool/RAGEN) or **Agent-Lightning** (RL-train an *existing* agent harness against `claw` with ~zero rewrite) for multi-turn agentic RL.

## Two hard gates BEFORE any RL run

1. **Harden the reward.** RL *will* find any seam in `code_exec.py` (empty-test passthrough, partial-credit gaming, editing the harness). Required before trusting the signal: immutable/read-only test files, sandboxed execution, and a **zero-and-exclude monitor** (mirror Ornith's layer-2). Also **fix the `llm_judge.py` silent-0.5-fallback bug** — under RL that's a direct reward-hacking vector, not just a measurement bug.
2. **Right-size expectations.** Qwen3.5 is already heavily RL'd; our Phase-3 finding is these bases *saturate* bounded suites. Expect **reliability / tool-discipline / format-adherence gains + modest pass@1**, not a capability jump; watch for diversity regression. If the reward curve is flat/hacked, that's itself the finding.

## Rig fit (2×96GB, 61GB RAM)

- GRPO concurrent residents: policy (QLoRA → tiny optimizer state), ref/KL, **rollout vLLM (the memory-hungry co-tenant)**. A 2B/4B QLoRA trainer fits one card with room; colocate + `vllm_enable_sleep_mode` offloads rollout VRAM when the trainer steps.
- **Bottleneck = rollout throughput, not optimizer memory** → our fast NVFP4/vLLM serving directly cuts step wall-clock. Caveat: NVFP4-*rollout* + bf16-*trainer* is a train/infer dtype gap to validate; simplest first pass = colocated vLLM at the trainer's dtype, NVFP4 as an optional rollout-speed lever.

## First steps (when we pull the trigger)

1. Harden `code_exec.py` + fix `llm_judge.py` 0.5-fallback (gate 1).
2. Stand up TRL GRPO on Qwen3.5-2B QLoRA, `code_exec` reward, colocated vLLM, tiny task set — get *one* clean reward curve.
3. Read the curve: real signal vs hacked vs flat. Only then scale tasks / consider veRL/agentic.

## Go/no-go

**GO on the scoped PoC** — on-thesis, cheap, and we uniquely own reward + fast rollout + LoRA plumbing. **Deliberate overnight+ commitment**, gated on grader hardening; not a SOTA run. Expansion to agentic/veRL gated on a non-hacked reward curve.
