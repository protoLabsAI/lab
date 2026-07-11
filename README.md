# protoLab — the heavy rig

Substrate #2 (Models) in the [protoLabs.studio](https://protolabs.studio) portfolio. **The quant + serving lab:** parity-verified FP8/quant models, serving findings from the heavy rig, and a *trustworthy* eval harness to back both. Model class of interest is small / on-device-capable (≤ ~35B, especially ≤9B) — the 2× RTX PRO 6000 Blackwell rig is the **forge**, not the inference target. Every finding is open-sourced as a pattern to study and steal.

Sibling: [`avaLab`](https://github.com/protoLabsAI/avaLab) on the A6000 — GGUF / llama.cpp / Ollama + ComfyUI. Cross-cutting work lives here. North star: [`FOCUS.md`](FOCUS.md). Studio overview: [`studio-brand/docs/explanation/portfolio.md`](https://github.com/protoLabsAI/studio-brand/blob/main/docs/explanation/portfolio.md).

## Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM each, 192 GB total), sm120
- vLLM 0.22.1 / CUDA 13 (torch 2.11+cu130, transformers 5.12); CUDA 12.8 toolkit also present
- CUDA graphs work on single GPU (37–470% speedup depending on model)
- **Replicas beat sharding when a model fits one card** — two single-GPU replicas (~6500 tok/s aggregate) beat DP+EP (3830) and TP=2 on PCIe. Don't shard what fits.
- TP=2 (when you must) is stable with `NCCL_P2P_DISABLE=1` (PCIe, no NVLink)
- `VLLM_USE_FLASHINFER_SAMPLER=0` required on sm120 (every model, 0.22.1)

## Layout

| Path | Purpose | State |
|------|---------|-------|
| `packages/lab-core/` | Pydantic models, GPU utils, paths | Publishable |
| `evals/` | Eval suite — claw-eval (submodule), custom suites, function-call, RAG, refusal | Strict + tested |
| `models/` | vLLM swap configs, MoE kernels, benchmark scripts | Mixed |
| `training/` | Fine-tuning workspace (LLaMA-Factory / TRL) | Loose |
| `experiments/` | Active research dirs (each with `RESULTS.md`) | Mixed |
| `infra/gateway/` | LiteLLM proxy + Langfuse observability | Operational substrate |
| `infra/{vllm,systemd}/` | service definitions | Operational substrate |
| `infra/{prometheus,exporters,crash-watch}/` | metrics, alerts, watchdogs | Operational substrate |

## What ships from here

- **Parity-verified quants on HuggingFace.** `Ornith-1.0` family — static FP8 of the models we actually serve, verified against the source before publishing (e.g. `Ornith-1.0-35B-FP8`: block-wise FP8, SSM kept bf16, 92.9% truly-fp8, coding/FC parity) — plus the `Ornith-1.0-9B-MTP` spec-decode head. The recipe + verification are the product. Org: [`protoLabsAI`](https://huggingface.co/protoLabsAI).
- **Serving findings as breakdowns.** Replicas-beat-sharding, CUDA graphs on Blackwell, the FlashInfer-on-sm120 recipe, MoE quant traps, spec-decode ladders — written up at [protolabs.studio](https://protolabs.studio).
- **The eval suite itself.** claw-eval is open; the custom + function-call + RAG suites are the lab's pattern.
- **The gateway.** `infra/gateway/` is the LiteLLM proxy every other studio service hits.

## Audience

Practitioners already operating in the LLM-agent space. If we have to explain context or tokens, this isn't the repo. The full audience filter lives in [`studio-brand/docs/reference/foundation.md`](https://github.com/protoLabsAI/studio-brand/blob/main/docs/reference/foundation.md) §3 — every breakdown we publish respects it. The open-source code itself has no filter — fork it, hack it to fit.

## Quick start

```bash
uv sync                                              # all workspaces
uv run pytest                                        # lab-core + evals tests

cd evals && ./run.sh profile --name quick --model local   # ~smoke eval, 1 trial
cd evals && ./run.sh claw --model local --tasks T02,T04    # specific claw tasks

bash models/vllm-swap.sh qwen-35b                    # swap the served vLLM model
bash models/speed-test.sh                            # decode tok/s from vLLM /metrics
uv run ruff check .                                  # lint
```

## Daily setup (dual GPU) — 2× Ornith-1.0-35B-FP8 replicas

All services are systemd, auto-start on boot. Daily driver = our own FP8 quant, one replica per card, gateway round-robin.

| GPU | Service | Model | Port | Notes |
|-----|---------|-------|------|-------|
| 0 | `vllm.service` | Ornith-1.0-35B-FP8 | :8000 | replica A, 256K, vision |
| 1 | `vllm-replica-b.service` | Ornith-1.0-35B-FP8 | :8003 | replica B, co-resident with Fish-TTS + embed |
| 1 | `protovoice-stack.service` | Fish S2-Pro TTS | :8092 | ~20 GB, lazy-load |
| 0/1 | `embed-{b,server}.service` | Qwen3-Embedding-0.6B | :8004/:8001 | doubled |

Gateway aliases (`infra/gateway/`): **`protolabs/smart`** round-robins the two Ornith replicas (~207 tok/s/req, ~6500 aggregate); **`protolabs/fusion`** is judged self-consistency (self-MoA) over `smart`; `protolabs/{reasoning,micro,nano}` are the other tiers. Eval judges target a local replica (`local` :8000/:8003), not the gateway round-robin.

## Standing baseline (claw-eval agent tasks)

**pass^3 dropped 2026-06-29** — single trial across full breadth discriminates models better than 3× repetition; numbers below are directional. Thinking-on, judged by `protolabs/reasoning`. Full methodology + artifacts in [`evals/baselines/`](evals/baselines/).

| Model | claw (mean) | FC | custom coding | Role |
|-------|:-----------:|:--:|:-------------:|------|
| **Ornith-1.0-35B-FP8** | **0.741** (35/35, 3×) | 93% | 0.925 | daily driver |
| Ornith-1.0-35B @ IQ2_M (2-bit, 12 GB) | 0.76 | 94% | 0.85 | low-bit-MoE point |
| Ornith-1.0-9B (bf16) | ~0.78 | 93% | 0.70 | edge-sized, best tool-caller |

**Headline cross-model finding:** a **2-bit 35B-A3B (12 GB) beats a bf16 9B (19 GB)** at smaller size — quant cost is asymmetric, it guts a small model but is near-free on a 3B-active MoE. On a fixed VRAM budget, spend it on a low-bit big model.

## Findings (the breakdown backlog)

- **Replicas beat DP+EP/TP=2 on PCIe.** When the model fits one card, two single-GPU replicas (~6500 tok/s aggregate) beat DP+EP (3830) and TP=2. Sharding tax > benefit.
- **FlashInfer-on-sm120 is cracked** (flashinfer 0.6.11 / CUDA 13) — the old "requires sm75+" wall fell to a 5-layer recipe ([`experiments/quantize/FLASHINFER-SM120-RECIPE.md`](experiments/quantize/FLASHINFER-SM120-RECIPE.md)). Opens FP8-KV cache, bf16 MoE, nano-LFM2.
- **CUDA graphs on Blackwell — 37–470% speedup.** MoE benefits most. Don't `--enforce-eager` on single GPU.
- **Spec-decode is a dense play.** 9B ladder (lossless): plain ~75 → MTP 121 → EAGLE-3 graft 138.8 tok/s. dFlash +43% single-stream on the 27B smart lane, but **MTP wins under concurrency** (C≥4). MoE: MTP is −11% (routing overhead > speculation), EAGLE worse.
- **MoE quant traps.** INT4 corrupts expert routing (keep MoE at FP8/BF16). `-O3`/torch-compile-L3 regresses MoE ~25%. `VLLM_USE_FLASHINFER_MOE_FP8=1` rejects Qwen block-wise FP8 — don't use.
- **NCCL_P2P_DISABLE=1 fixes TP=2 corruption on PCIe Blackwell.** Root cause: ACS-enabled PCIe bridges corrupt P2P during CUDA graph replay.
- **Self-MoA beats mixed-MoA (reproduced).** `protolabs/fusion` (judged self-consistency over one strong model): 44.8% blind-rank win vs solo 24.1%; a `diverse` panel scored *worse* (31%) — diversity is a tax. ~3× tokens → hard prompts only, never default traffic.
- **Capability cliff at 4B → 2B.** Sub-4B can't do agentic tool use.

Full ops detail and reproduction commands in [`CLAUDE.md`](CLAUDE.md).

## Status

North star is the quant + serving lab ([`FOCUS.md`](FOCUS.md)). Active: `evals/` (single-trial, thinking-on, one metric per suite), `models/`, `experiments/{quantize,audio-tags,agentworld,dflash,eagle3,mtp,embedding-bench,rag-bench,context-1,vllm-dashboard}/`, `infra/`. Stopped (archived to `/mnt/data/lab-archive/`, recoverable): the brand-pivot side-bets and the "eval every new model" metric zoo. `audio-tags` survives as the standalone brand exemplar. Image/voice work lives on avaLab.

## Secrets

Managed by [Infisical](https://secrets.proto-labs.ai), self-hosted. Zero secrets in this repo. Gateway `start.sh` authenticates via Machine Identity and injects env vars at runtime.
