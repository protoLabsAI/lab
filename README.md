# protoLab — the heavy rig

Substrate #2 (Models) in the [protoLabs.studio](https://protolabs.studio) portfolio. FP8 / vLLM lab on 2× RTX PRO 6000 Blackwell. State-of-the-art models, prosumer hardware, every finding open-sourced as a pattern to study and steal.

Sibling: [`avaLab`](https://github.com/protoLabsAI/avaLab) on the A6000 — GGUF / llama.cpp / Ollama + ComfyUI. Cross-cutting work lives here. Studio overview: [`studio-brand/docs/explanation/portfolio.md`](https://github.com/protoLabsAI/studio-brand/blob/main/docs/explanation/portfolio.md).

## Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM each, 192 GB total)
- CUDA 12.8, driver 595.x
- CUDA graphs work on single GPU (37–470% speedup depending on model)
- TP=2 stable with `NCCL_P2P_DISABLE=1` (PCIe, no NVLink)

## Layout

| Path | Purpose | State |
|------|---------|-------|
| `packages/lab-core/` | Pydantic models, GPU utils, paths | Publishable |
| `evals/` | Eval suite — claw-eval (submodule), custom suites, function-call, RAG, refusal | Strict + tested |
| `models/` | vLLM configs, MoE kernels, benchmarks | Mixed |
| `training/` | Fine-tuning workspace | Loose |
| `experiments/` | Active research dirs + `PARKED.md` memos | Mixed |
| `infra/gateway/` | LiteLLM proxy + Langfuse observability | Operational substrate |
| `infra/vllm/` | systemd service definitions | Operational substrate |
| `infra/prometheus/` | Metrics + alert rules | Operational substrate |

## What ships from here

- **Findings as breakdowns.** CUDA graphs on Blackwell, NCCL_P2P_DISABLE fixing TP=2 PCIe corruption, FP8 KV cache broken on sm120, INT4 routing instability on MoE — written up at [protolabs.studio](https://protolabs.studio).
- **FP8 quants on HuggingFace.** Qwen3.6 family, published via `experiments/quantize/`. Org: [`protoLabsAI`](https://huggingface.co/protoLabsAI).
- **The eval suite itself.** claw-eval is open. The custom + function-call + RAG suites are next.
- **The gateway.** `infra/gateway/` is the LiteLLM proxy every other studio service hits.

## Audience

Practitioners already operating in the LLM-agent space. If we have to explain context or tokens, this isn't the repo. The full audience filter lives in [`studio-brand/docs/reference/foundation.md`](https://github.com/protoLabsAI/studio-brand/blob/main/docs/reference/foundation.md) §3 — every breakdown we publish respects it.

The open-source code itself has no filter — fork it, hack it to fit. No feature requests from anyone who hasn't worked through the docs first.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync                                                # all workspaces
uv run pytest                                          # lab-core + evals tests
uv run proto-eval claw --model local --tasks T02,T04   # claw-eval tasks
uv run models --gpu single                             # model inventory
bash models/vllm-swap.sh qwen-27b-int4                 # swap vLLM model
uv run ruff check .                                    # lint
```

## Daily setup (dual GPU)

| GPU | Service | Model | Port | tok/s |
|-----|---------|-------|------|-------|
| 0 | `vllm.service` | Qwen3.6-27B-FP8 (thinking, 225K) | :8000 | ~73.6 (+MTP) |
| 1 | `vllm-fast.service` | Qwen3.6-35B-A3B-heretic-FP8 (instruct, 262K) | :8002 | 199 |

Gateway aliases via `infra/gateway/`: `protolabs/smart` → 27B, `protolabs/fast` → 35B MoE.

## Model leaderboard (claw-eval agent tasks)

Last refreshed 2026-04-29. Pass^3 (3 trials, all-pass required). Single-GPU configs unless noted.

| Rank | Model | tok/s | pass^3 | Avg score | Config |
|------|-------|:-----:|:------:|:---------:|--------|
| 1 | Qwen 35B MoE BF16 TP=2 | 170 | 3/4 | 0.80 | TP=2, 250K ctx |
| 2 | Qwen 27B INT4 | 44 | 3/4 | 0.79 | Single GPU, 160K ctx |
| 3 | Qwen 122B INT4 (1 GPU) | ~30 | 3/4 | 0.78 | enforce-eager, 64K |
| 4 | OmniCoder 9B | 92 | 2/4 | 0.76 | Single GPU, 262K ctx |
| 5 | Llama 70B AWQ | 38 | 1/4 | 0.65 | Creative writing only |

Cloud comparison (same eval, run 2026-03): GLM 5 Turbo 0.85, Sonnet 4.6 0.85, Opus 4.6 0.84.

## Findings (the breakdown backlog)

- **CUDA graphs on Blackwell — 37–470% speedup.** MoE models benefit most (3B active → 170 tok/s).
- **INT4 on dense models — no quality loss vs BF16.** Use GPTQ-Int4 for dense, BF16 for MoE.
- **MoE INT4 instability.** Quantization corrupts expert routing. Keep MoE at BF16.
- **NCCL_P2P_DISABLE=1 fixes TP=2 corruption on PCIe Blackwell.** 122B INT4 23 → 122 tok/s (5.3×); 35B MoE 22 → 205 tok/s (9.3×). Root cause: ACS-enabled PCIe bridges corrupt P2P during CUDA graph replay.
- **MTP speculative decoding.** +47% on 27B FP8, +32% on 27B INT4, +22% on 9B; **−11% on 35B MoE** (routing overhead beats speculation savings).
- **FP8 KV cache broken on sm120.** FlashInfer attention assumes sm75–sm90. Re-verified 2026-05-03. `VLLM_USE_TRITON_FP8_GEMM=1` fixes matmul, not attention.
- **Inference draws 300–340 W per GPU regardless of power limit.** MoE only 88–96 W per card at 600 W limit — not power-bound.

Full ops detail and reproduction commands in [`CLAUDE.md`](CLAUDE.md).

## Status (2026-05-22)

Studio refocused on the brand. ORBIS retired as a product, coding-agent work out of scope. Side bets parked under `experiments/*/PARKED.md`. Active here: `evals/`, `models/`, `experiments/{audio-tags,context-1,quantize,embedding-bench,gemma4-eval,proto-bench,rag-bench,vllm-dashboard}/`, `infra/`.

## Secrets

Managed by [Infisical](https://secrets.proto-labs.ai), self-hosted. Zero secrets in this repo. Gateway `start.sh` authenticates via Machine Identity and injects env vars at runtime.
