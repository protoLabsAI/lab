# protoLab — Focus

**North star: the quant + serving lab.** We publish parity-verified FP8/quant models, we publish serving findings from the heavy rig, and we run a *trustworthy* eval harness to back both. Model class of interest: small / on-device-capable (≤ ~35B, especially ≤9B). The heavy rig (2× RTX PRO 6000 Blackwell) is the **forge**, not the inference target.

## What we do (and ship)

1. **Quant quality** — static FP8 (and friends) of models we actually serve, **parity-verified** against the source before publishing to [`protoLabsAI`](https://huggingface.co/protoLabsAI). Recipe + verification are the product, not just the weights. (e.g. `Ornith-1.0-35B-FP8`: block-wise FP8, SSM kept bf16, 92.9% truly-fp8, coding/FC parity.)
2. **Serving findings** — what actually makes models fast/correct on this hardware. (e.g. **replicas beat DP+EP/TP=2 on PCIe**; CUDA graphs on Blackwell; MoE quant traps; NCCL/PCIe.) Each is a reusable, reproducible finding.
3. **Trustworthy evals** — `evals/` as the open-source pattern. Reasoning models evaluated **thinking-on**; one coherent metric per suite; standing baselines; no silent failures.

## What we stopped (2026-06-27)

Archived to `/mnt/data/lab-archive/` (recoverable): the 13 brand-pivot side-bets (companion-stack, voice-agent, salm-duplex, rlm, agent-lightning, qwen3-omni, stt-whisper, tts-compare, image-gen-eval, flux2, pixel-gen, ltx-video, proto-bench) + diffusion side-bets (diffusiongemma, diffusion-cli-tools). Image/voice work lives on avaLab. **We stop: one-off "eval every new model" runs, the metric zoo, breadth for its own sake.**

## Eval suite — stabilization plan (Phase 2, in progress)

The current harness produces noise we reverse-engineer. Fixing, in priority order:

1. ~~**No silent failures.**~~ ✅ **FIXED (2026-06-27).** Root cause: the `X-Health-Check` probe POSTed an empty body to validating endpoints (`/kb/search`, `/contacts/search`) → FastAPI 422 → harness marked the service permanently unhealthy → every kb/contacts task (~10/run) silently failed. Fix: health probe short-circuits to a 200 liveness response (claw-eval `mock_services/_base.py`). Verified: kb/contacts tasks now run + score. **Next**: make the runner *report* harness-errored tasks distinctly from model-scored ones (so a run says "33 scored, 2 harness-errored", never a silent average).
2. **One metric per suite.** Kill the `passed` vs `task_score` vs `pass^3` ambiguity — pick one primary number (claw: mean task_score; FC: pass rate; coding: avg_score) and report it consistently.
3. ~~**Standing baselines.**~~ ✅ DONE (2026-06-27) — `evals/baselines/README.md` + first Ornith-35B-FP8 baseline recorded (claw **0.672** 35/35 clean / coding **0.925** / FC **93%**, reasoning judge). Re-run on every methodology change.
4. ~~**Consolidate runners.**~~ ✅ DONE (2026-06-27) — kept `claw`/`custom`/`function-call`/`rag` (+ `profile`/`compare`); archived `wildbench`/`refusal`/`inspect`/`general` to `/mnt/data/lab-archive/runners-2026-06-27/`. rag's judges already route through the fixed `LLMJudge` (4096).
5. ~~**Pinned judge.**~~ ✅ DECIDED (2026-06-27): **`protolabs/reasoning`** (independent cloud reasoning model via gateway) for **baselines**; local judge OK for everyday/relative runs. Never self-judge a baseline. Silent-0.5 fallback hardened in `llm_judge.py`.

## Current production (see [memory] / CLAUDE.md)

- Smart lane = **2× `Ornith-1.0-35B-FP8` replicas** (systemd `vllm` + `vllm-replica-b`), gateway round-robin (`least-busy`). Embeddings doubled + balanced.
- Replicas > sharding on PCIe — never TP/EP for a model that fits one card.
