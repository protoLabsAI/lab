# agentworld — language world model as a synthetic agentic environment

`Qwen/Qwen-AgentWorld-35B-A3B` is a **world model, not an agent**: a Qwen3.5-35B-A3B-Base
fine-tune (CPT→SFT→RL/GSPO, 10M+ trajectories) that **simulates** agentic environments —
given an action + interaction history it predicts the *next environment observation* across
seven domains (MCP, Search, Terminal, SWE, Android, Web, OS). Same arch/size as our
`Ornith-1.0-35B` daily driver, so every serving/quant finding transfers. Apache-2.0.

Full framing + the four plays: [`../BACKLOG.md`](../BACKLOG.md) §3.

## The central question

A world model gives a **plausible-but-not-verifiable** reward/observation — the philosophical
opposite of the `game-rlvr` substrate's **verifiable, deterministic** reward. Before trusting
any eval or RL loop built on it, we measure one thing: **how far does its simulated environment
diverge from the real one?** AgentWorldBench self-reports wildly domain-dependent fidelity
(OS 65.92 vs Search 36.69 / 100). We replicate that on *our* tasks.

## Plays (ranked, see BACKLOG §3)

1. **FP8 quant publish** — near-zero-cost, same arch as Ornith. Runs independent of research.
2. **Distill a ≤4B single-domain simulator** — Docker-free CI-time mocking. Gated by #3.
3. **Differential-fidelity study (brand piece)** — sim vs real across domains. **Gates #2/#4.**
4. **Adversarial env perturbation** — chaos-monkey envs to stress Ornith (T28 safety gap).

## First move: the fidelity probe (play #3, minimal form)

We already have **free ground truth**: yesterday's Ornith-9B claw run left real
`(command → {exit_code, stdout, stderr})` transcripts for the T100–T104 terminal/SWE tasks at
`evals/results/ornith-9b_20260628_021428/`. Each `tool_dispatch` record is one real
sandbox step — no Docker re-run needed.

`probe_fidelity.py`:
1. Parses a claw `.jsonl` trace → ordered list of real `(command, observation)` steps.
2. Replays each command into AgentWorld as a Terminal world-model turn, feeding the *real*
   prior observations as history (teacher-forced, so errors don't compound).
3. Diffs simulated vs real observation: exact stdout match, exit-code match, normalized
   line-overlap, and length ratio. Emits per-step + per-task fidelity.

One divergence number per task decides whether the synthetic-environment idea is real for our
suite before any wiring.

## Status

- [x] Model downloaded → `models--Qwen--Qwen-AgentWorld-35B-A3B` (21 shards, 65G)
- [x] Ground-truth tapes located (T100–T104, Ornith-9B 2026-06-28 run)
- [x] `probe_fidelity.py` written, MCP-framed, official parser ported
- [x] Canonical prompts pinned in `prompts/` (Terminal + MCP from the repo)
- [x] Served AgentWorld FP8 on GPU 1 :8010 (replica-b stopped — **restore when done**)
- [x] Ran probe on T101 (confounded, discarded) + T103 (v1) — see [RESULTS.md](RESULTS.md)
- [x] **Key finding:** simulates a sandbox's *shape*, not its *state* (fixture content unknowable)
- [x] **RL synthesis written** → [PROPOSAL.md](PROPOSAL.md): AgentWorld as the imagination buffer for
      Ornith-style self-scaffolding RL (sim shapes the scaffold; real sandbox stays the verifier)
- [ ] v2: LLM-judge metric (Format/Consistency/Realism on Ornith :8000) + sweep T100–T104
- [ ] (parallel, independent) FP8 quant publish via `experiments/quantize/`
- [ ] Scaffold-transfer probe from PROPOSAL.md (does a sim-matured scaffold raise real pass-rate?)

## Serving (needs one free GPU)

Both GPUs currently run the Ornith replicas (`vllm.service` :8000, `vllm-replica-b.service`
:8003). To probe, free GPU 1 and serve single-card:

```bash
sudo systemctl stop vllm-replica-b.service        # free GPU 1
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve Qwen/Qwen-AgentWorld-35B-A3B \
  --host 0.0.0.0 --port 8010 \
  --quantization fp8 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code \
  --gpu-memory-utilization 0.62
```

- `--quantization fp8` is **required**, not optional: this is a 35B-A3B MoE and on sm120 the
  **bf16** MoE path routes to flashinfer-cutlass and crashes (CLAUDE.md Blackwell constraints).
  On-the-fly FP8 takes the Triton fused-MoE path (works) and shrinks weights ~69→35GB so the
  model co-resides with Fish TTS (~20GB) + embed-A (~2GB) still on GPU 1.
- `--language-model-only` strips the vision tower (config is `…ForConditionalGeneration`).
- `VLLM_USE_FLASHINFER_SAMPLER=0` is the standing sm120 requirement.
- Do **not** add `-O3` (regresses MoE ~25%). util 0.62 measured against free VRAM after replica-b stops.

```bash
python probe_fidelity.py \
  --trace ../../evals/results/ornith-9b_20260628_021428/ornith-9b_26-06-28-04-14/T101_wal_recovery_913fa180.jsonl \
  --endpoint http://localhost:8010/v1 --model Qwen/Qwen-AgentWorld-35B-A3B
```
