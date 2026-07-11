# Agentic-Coding-RL — Ornith harness × Coder-Next coding

**One-liner:** take the best open coding base (Qwen3-Coder-Next, 80B/3B-active), distill Ornith's self-scaffolding *harness behavior* into it, then RL-polish under our verifiable reward, and ship the merged model as a one-card NVFP4 agentic coder.

## Why this shape (the load-bearing thesis)

Two capabilities acquired at *different* stages:
- **Coding prowess = pretraining-limited.** Coder-Next's base is code-specialized (SWE-bench 70.6, beats the 480B flagship). Ornith's Qwen3.5 base is general → a lower coding ceiling you can't RL past. → **Start from Coder-Next.**
- **Self-scaffolding harness = RL/distill-addable.** It's a post-training behavior — exactly what Ornith's RL produced and a raw coder lacks. → **Add it to Coder-Next.**

We can't get Ornith's recipe/data (undisclosed), but **we have the Ornith weights** → use Ornith as the *teacher* of harness behavior, Coder-Next as the *coding-strong student*.

## Phases

### Phase 0 — Baseline (IN PROGRESS)
Coder-Next-NVFP4 on our eval, judge pinned to live `:8000/local`:
- **coding** (custom suite, code-exec) → the ceiling (expect strong)
- **function_call** + **claw** (agentic tool-use) → the *harness gap* (expect weak — this is what we lift)
Serves at `:8005`, 181 tok/s one-card. This before-number is the whole yardstick.
> Gotcha hit: coding runner defaults `--max-tokens 32768` == served `--max-model-len` → 400. Use `--max-tokens 8192`, or serve Coder-Next at 64K+.

### Phase 1 — Harness distillation (SFT)
1. Serve **Ornith** (teacher) — it emits self-scaffolding trajectories (plan → tool calls → solution) by construction.
2. Generate trajectories on a coding+agentic task set (SWE-style + our claw/code tasks).
3. **SFT (QLoRA) into Coder-Next** — teach the *scaffolding format/behavior*, not the code (it already codes).
Output: Coder-Next that self-scaffolds. Watch the coding suite the whole time for regression.

### Phase 2 — RL-polish (GRPO)
GRPO/QLoRA on the SFT'd Coder-Next, **reward = hardened `code_exec.py`** (+ scaffolding-step shaping). TRL `GRPOTrainer` (or Unsloth for qwen3_next QLoRA efficiency), **colocated NVFP4 vLLM rollout** (sleep-mode) — 3B-active + NVFP4 makes rollouts cheap, which is RL's real bottleneck. Skip MTP (open +76% regression on qwen3_next).

### Phase 3 — Merge → quant → gate → ship
- Merge QLoRA into a **bf16** copy (layer-wise, streams), NOT the 4-bit.
- Re-quantize merged → **NVFP4** (~40 GB, `device_map=auto` in VRAM, bypasses the 61 GB RAM wall).
- Serve one card; **re-run the Phase-0 eval**. Ship iff **agentic (claw/FC) up meaningfully AND coding held.**

## Gate 1 — HARDEN THE REWARD (prerequisite, do before Phase 2)
RL finds every seam. **Due-diligence delta (see `RESEARCH.md`): the eval grader and the RL
reward are two different jobs.** `code_exec.py`'s partial credit is right for EVAL (it makes the
suite discriminate) but is a gaming gradient under RL — the field converged on **sparse binary,
all-tests-pass, no partial credit, no reward model** (DeepSWE, Kimi-Dev). So Gate 1 = a *separate*
hardened reward (`code_reward.py`), leaving `code_exec.py` untouched for eval.

The reward-hacking catalog to defend against (arxiv 2604.15149, verified): overwrite/delete tests,
monkeypatch scoring fns, replace asserts with passing prints, `exit(0)` before grading, hardcode
expected outputs. Our live seam: `code_exec.py:70` execs the solution into the **same `globals()`**
as the test battery — enables every one.

Requirements:
- **Sparse binary reward** (all hidden tests pass → 1.0, else 0.0) — NOT partial credit.
- **Namespace/interpreter isolation** — solution execs in its own dict; grading uses stdlib refs
  captured *before* the solution runs; tests run with a clean pre-snapshot `__builtins__`.
- **Unforgeable grade channel** — results emitted over a private fd with a nonce sent out-of-band
  (not argv/env), argv/env scrubbed before the solution runs, so monkeypatching stdout can't fake a pass.
- **Read-only hidden tests** never placed in the model's cwd or prompt (data-pipeline property).
- **Zero-and-exclude monitor** (Ornith layer-2): static red-flag scan + dynamic tamper detection →
  gaming returns reward 0 **and** an `exclude` flag to drop the trajectory from advantage.
- **DeepSWE Compact-Filtering** style masking (timeout/step-limit → exclude) — but A/B it (SWE-Master
  found the same masking hurt).
- **`llm_judge.py`**: add a strict `raise_on_error` mode so an RL caller gets a sample-excluded
  signal instead of a farmable 0.5 (eval default unchanged — loud 0.5 stays for measurement).
See `experiments/rlvr-poc/RESEARCH.md` for framework/memory details and `RESEARCH.md` for prior art.

## Rig fit
- QLoRA on 80B/3B-active: 4-bit frozen base (~40 GB) + tiny LoRA; forward/backward only through ~3B active → trains *lighter* than a dense 80B. Fits our 2×96 GB.
- 2-card layout: NVFP4 rollout (sleep-mode) on one card + QLoRA trainer on the other, or colocate.
- Tooling: Unsloth supports `qwen3_next` (the quants are theirs); TRL GRPO for the RL loop.

## Success metric & risks
- **Metric:** Phase-0 → Phase-3 delta on our eval — claw/FC **up**, coding **held**. That delta *is* the result.
- **Risks:** reward hacking (Gate 1); catastrophic forgetting of coding during SFT/RL (monitor coding every checkpoint); thin coding headroom (frame the win as *agentic*, not codegen).
- **Honest prior:** biggest, cleanest gains are agentic reliability / tool-discipline / scaffolding — exactly the dimension Coder-Next is weak and Ornith is strong. That's the whole bet.

## First buildable step
**Gate 1 — reward hardening** (`code_exec.py` sandbox + immutable tests + zero-and-exclude monitor; fix `llm_judge.py` 0.5-fallback). It's our own code, buildable now, and blocks Phase 2. Everything else (teacher-traj gen, GRPO wiring) sits on top.
