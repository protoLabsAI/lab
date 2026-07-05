# STATUS — agentic-coding-rl

## Done & verified
- **Phase 0 baseline** — `BASELINE.md` (Coder-Next-NVFP4: coding 67%, FC 94%, claw 0.485).
- **Due diligence** — `RESEARCH.md` (24 sources, 25 claims verified). Flipped 3 plan choices:
  sparse-binary RL reward, NVFP4-is-ship-not-train, minimal cold-start over big SFT.
- **Gate 1 — hardened reward** — `evals/graders/code_reward.py`. Sparse binary, interpreter
  isolation, unforgeable nonce grade channel, zero-and-exclude monitor. `test_code_reward.py`
  = 11/11 vs the attack catalog (incl. 2 runtime-only proofs). `llm_judge.py` strict mode added.
- **Phase 1 plumbing (GPU-free)** — verified via `test_wiring.py`:
  - `rl_dataset.py` — loads `code_exec`-graded coding YAML → RL task records; prompt kept
    free of test asserts (hidden-tests data-pipeline property). Currently **14 tasks**.
  - `reward_fn.py` — TRL `GRPOTrainer`-compatible reward callable over `code_reward.score()`.
    Exclude policies: `penalize` (TRL-native, gamed→−1.0) and `mask` (gamed→0.0 + `fn.last.
    exclude_mask` for a masking trainer). Handles str + chat-format completions.

## Next (GPU-gated — the deliberate overnight commitment, per rlvr-poc go/no-go)
1. **Expand the task set.** 14 is too thin for a reward curve; rlvr-poc wants a few hundred–few
   thousand. Lever: `evals/tasks/coding/generators/gen_hard_v2.py` (check if procedural vs
   model-gen before scaling). Each task needs an executable hidden-test battery.
2. **GRPO trainer** (`train_grpo.py`, NOT yet written — don't ship it unrun). TRL `GRPOTrainer`,
   policy = Qwen3.5-2B/4B QLoRA (start small/dense; note Unsloth warns against 4-bit QLoRA on
   Qwen3.5 MoE — RESEARCH.md §4), colocated vLLM sleep-mode rollout, `reward_funcs=[reward_fn]`.
   For true zero-and-exclude, subclass GRPOTrainer to read `reward_fn.last.exclude_mask` and mask
   per-sample loss; until then use `exclude_policy="penalize"`. RL in bf16/NF4 w/ **frozen MoE
   router** (Nemotron lever), requant NVFP4 only at ship.
3. **Read the curve** (rlvr-poc gate): real signal vs hacked vs flat. `reward_fn.last.gaming_rate`
   is the live reward-hacking monitor. Only scale / go agentic (veRL) on a non-hacked curve.

## Ordering decision still open (RESEARCH.md)
Pure-RL (DeepSWE R1-Zero) vs minimal ~5k Agentless cold-start (Kimi-Dev) vs big teacher SFT then
GRPO (Open-SWE). Evidence contradictory across bases; needs an A/B on Coder-Next. Gate 1 + the
reward plumbing are prerequisites for all three, so they were built first.
