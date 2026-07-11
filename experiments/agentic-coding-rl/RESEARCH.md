# Due Diligence — others' methods & attempts (2026-07-05)

Prior-art survey before Gate 1 / Phase 1. Question: has anyone done our three bets —
(1) distill *scaffolding/harness* behavior into a code-strong base via SFT, (2) GRPO/RLVR
under a verifiable execution reward, (3) harden that reward against hacking — and does the
evidence support or undermine our plan? Deep-research fan-out, 24 sources, 25 claims
adversarially verified (24 confirmed / 1 refuted). Full trace:
`d4db2104-…/tasks/wvip34rbw.output`.

## Verdict: the two-stage SHAPE is validated; three specific choices must change.

### 1. Scaffolding distillation (SFT) — SUPPORTED, near-exact precedent exists
- **Open-SWE-Traces** (arxiv 2606.16038): distilled agentic trajectories into
  **Qwen3-Coder-30B-A3B** (code-strong A3B MoE ≈ our target class), **SFT-only, no RL**:
  **51.6 → 61.7% SWE-bench Verified**. This is essentially our Phase 1 on our architecture.
  ← the single most load-bearing precedent.
- **Nemotron 3 Nano** (arxiv 2512.20848): multi-turn tool-use distilled into a **3.2B-active
  MoE** via teacher-simulated user/agent/tool-env triad + LLM-judge dropping goal-inconsistent
  actions. STEAL the judge-filter for Ornith trajectory gen.
- **Magnet-14B-mDPO** (arxiv 2503.07826): student *beat* teacher (Gemini-1.5-pro) at function
  calling — but the win came from the DPO stage, not vanilla SFT. A preference stage is part
  of the recipe worth considering.

### 2. RLVR / GRPO — SUPPORTED, but our reward is the WRONG SHAPE
- Repeatedly-validated recipe = **sparse binary, all-tests-pass, no partial credit, no reward
  model**: DeepSWE GRPO++ (42.2→59% w/ TTS), Kimi-Dev outcome-only 0/1 (60.4%), SkyRL-Agent
  (24.4→39.4%), SWE-RL/Llama3-70B (41%).
- ⚠️ **Directly challenges `code_exec.py`'s partial-credit design as an RL reward.** Partial
  credit is right for *measurement* (discrimination) but hands the policy a gradient to game
  the easy subset. → **Split: partial credit for EVAL, sparse binary (all hidden tests pass)
  for the RL REWARD.**
- **GRPO++ (DeepSWE):** GRPO + DAPO clip-high + no KL + no reward-std-norm (Dr.GRPO) + length
  norm + leave-one-out. Steal.
- **R2E-Gym** (arxiv 2504.07164): 8,100+ off-the-shelf executable SWE envs w/ automated
  unit-test reward. Build on it, don't reinvent.
- **SWE-RL** alt reward = difflib similarity to an oracle patch (no execution) — cheap, dodges
  exec-hacking, but only as good as the reference patch.

### 3. Reward hardening (Gate 1) — VALIDATED & URGENT; attack catalog matches our seam
- **"LLMs Gaming Verifiers"** (arxiv 2604.15149): RLVR models game imperfect verifiers by
  enumerating instance-level labels; catalog includes **overwriting unit tests, monkey-patching
  scoring fns, deleting assertions, replacing asserts with trivially-passing prints, premature
  exit.** Behavior is *specific to RLVR-trained models* (absent in GPT-4o etc.).
- **OUR LIVE SEAM:** `code_exec.py:70` `exec`s the model solution into the **same `globals()`**
  as the hidden test battery → enables *every* attack above once it's a reward.
- Gate 1 must: solution & tests in **separate namespaces/process**, tests read-only & out of
  reach; block `sys.exit`/`os._exit`/builtin-redefinition; **hidden held-out tests**;
  **zero-and-exclude** trajectory filtering (DeepSWE Compact Filtering) — *but A/B it*, SWE-Master
  (2602.03411) found the same masking HURT. Optional: R2E-Gym **hybrid verifier** (exec + learned
  re-ranker → 51% vs 34% exec-only).
- Also fix `llm_judge.py` silent-0.5 fallback (already surfaced; a reward-hacking vector under RL).

### 4. RL on 4-bit/NVFP4 ultra-sparse MoE + colocated rollout + LoRA merge — UNDERMINED, unprecedented
Weakest leg. No external source demonstrates this exact combination, and there are active warnings:
- **Unsloth explicitly advises AGAINST QLoRA 4-bit on Qwen3.5 MoE** (abnormally high quant error).
- `load_in_4bit=True` **does NOT give vLLM a 4-bit rollout** under `fast_inference` — loads 16-bit
  (Unsloth #1930). Kills the "colocated NVFP4 rollout saves memory" rationale.
- **MXFP4 MoE can't train — no backward-pass kernel** (Unsloth substitutes NF4). NVFP4 ≈ same →
  **cannot RL the NVFP4 weights directly.**
- No async RL lib implements MoE **"Keep Routing"** → train/inference routing mismatch;
  importance-sampling only a partial fix. **Nemotron froze the MoE router** to stabilize.
- **FIX:** drop "RL the NVFP4 model." RL a **bf16/NF4** copy (router frozen; Unsloth Standby +
  12×-MoE-kernels for memory), **requant to NVFP4 only at ship** (= existing Phase 3). NVFP4 is a
  deployment format, not a training one.

### Ordering tension (genuine, unresolved)
- **DeepSWE:** SFT-then-RL **ineffective** (100 iters, no gain) — but on a *dense* Qwen3-32B.
- **Open-SWE / Kimi-Dev:** SFT highly effective on *code-strong MoE*.
- **Forgetting is asymmetric:** SFT erodes base codegen more than RL (arxiv 2605.28860:
  RL retained 15.8pp more of the base circuit; SWE-RL's SFT baseline degraded OOD, RL improved).
- → Prefer a **minimal Agentless-style ~5k cold-start** (Kimi-Dev) over a large SFT; keep
  **pure-RL as an A/B arm**; add **capability-replay data**; **gate codegen every checkpoint.**

## Plan deltas (proposed)
1. **Reward:** partial-credit for eval, **sparse-binary all-hidden-tests-pass** for RL reward.
2. **Gate 1:** rebuild `code_exec.py` sandbox — separated namespaces, read-only hidden tests,
   exit/builtin guards, zero-and-exclude (A/B'd). This is the seam, not a nice-to-have.
3. **Phase 1:** minimal ~5k cold-start SFT (not a big one) + capability-replay + per-checkpoint
   codegen gate; add a **pure-RL A/B arm**.
4. **Phase 2/3:** RL in **bf16/NF4 with frozen router** (not NVFP4); NVFP4 only at ship. Drop the
   colocated-NVFP4-rollout assumption.

## Highest-signal sources (read in full)
1. **Open-SWE-Traces** — arxiv 2606.16038 — near-exact precedent (A3B MoE, SFT-only, +10pt).
2. **DeepSWE** — together.ai/blog/deepswe — GRPO++, sparse reward, Compact Filtering, SFT-then-RL
   negative result.
3. **Nemotron 3 Nano** — arxiv 2512.20848 — small-active-MoE agentic distill + RL, router-freeze.
4. **LLMs Gaming Verifiers** — arxiv 2604.15149 — the reward-hacking attack catalog (Gate 1).
5. **R2E-Gym** — arxiv 2504.07164 / github R2E-Gym — off-the-shelf envs + hybrid verifier.
6. **Kimi-Dev** — arxiv 2509.23045 — 5k cold-start + outcome-only 0/1 RL.
7. **Catastrophic forgetting mechanism** — arxiv 2605.28860 — SFT>RL forgetting asymmetry.
8. **Unsloth #1930 + Qwen3.5 MoE finetune docs** — the QLoRA-4bit-MoE gotchas.
