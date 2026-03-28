# CLAUDE.md — protoLabs AI Lab

Monorepo for model evaluation, training, inference infrastructure, and ML experiments.

## Structure

- `packages/lab-core/` — Shared Pydantic models, GPU utils (strict, tested, publishable)
- `evals/` — LLM eval suite: claw-eval, custom suites, function-call, RAG (strict, tested)
- `models/` — Model inventory, vllm-swap.sh, benchmarks
- `training/` — Fine-tuning workspace (LLaMA-Factory configs, datasets)
- `experiments/` — ML experiments (ltx-video, flux2, quantize, pixel-gen, stt-whisper, voice-agent)
- `infra/` — Prometheus exporters, vLLM systemd, gateway configs (gateway runs on ava node)

## Using uv

```bash
uv sync                                    # sync all workspaces
uv run pytest                              # run tests (lab-core + evals)
uv run proto-eval claw --model local       # run evals CLI
uv run models --gpu single                 # show model inventory
uv run ruff check .                        # lint everything
```

## Running Models

```bash
bash models/vllm-swap.sh qwen-35b           # speed king MoE FP8 official, 262K (180 tok/s)
bash models/vllm-swap.sh qwen-9b-fp8        # on-the-fly FP8 (140 tok/s)
bash models/vllm-swap.sh qwen-4b-fp8        # on-the-fly FP8 edge (140 tok/s)
bash models/vllm-swap.sh qwen-27b-int4      # daily driver, agentic (53 tok/s)
bash models/vllm-swap.sh qwen-27b-int4-mtp  # daily driver + MTP, chat/creative (70 tok/s)
bash models/vllm-swap.sh qwen-4b-int4       # edge deploy, fastest absolute (297 tok/s)
bash models/vllm-swap.sh qwen-4b            # LoRA base bf16 (155 tok/s)
bash models/vllm-swap.sh qwen-122b-fp8      # quality ceiling FP8 official TP=2 (112 tok/s)
bash models/vllm-swap.sh qwen-122b-int4     # quality ceiling INT4 TP=2 (122 tok/s)
bash models/vllm-swap.sh qwen-27b-fp8-tp2   # FP8 official TP=2 (70 tok/s, 131K)
```

## Speed Testing

```bash
bash models/speed-test.sh           # 5 runs on current model (800 tok gen)
bash models/speed-test.sh 10        # 10 runs
bash models/speed-test.sh 3 short   # 3 short runs (200 tokens)

# A/B compare baseline vs optimized config
cd evals && bash run-ab-speed.sh qwen-4b-int4 5
```

Reports decode tok/s (1/TPOT), wall tok/s, TTFT, and TPOT from vLLM's `/metrics` endpoint — not wall-clock estimation.

### Optimization Flags (`-opt` configs)

Suffix any config with `-opt` to enable P1+P2 flags:
- `--async-scheduling` — overlap scheduling with execution
- `--enable-prefix-caching` — reuse KV cache for repeated prefixes
- `--performance-mode interactivity` — auto-tune scheduler for latency
- `--kv-cache-dtype fp8` — halve KV cache memory, double context capacity

**Measured impact (single-request, P1+P2 only):** minimal (+1-3% tok/s). Real wins are under concurrent load and multi-turn (prefix caching). FP8 KV doubles context capacity.

### MTP Speculative Decoding (`-mtp` configs)

Native Qwen3.5 Multi-Token Prediction — big speed gains on dense models:

| Model | Baseline | + MTP | Gain | Tool Calling |
|-------|:--------:|:-----:|:----:|:------------:|
| **27B INT4** | 53 tok/s | **70 tok/s** | **+32%** | Works, but T08 quality regresses |
| **9B** | 92 tok/s | **112 tok/s** | **+22%** | Works, no quality loss |
| **35B MoE** | 171 tok/s | 153 tok/s | -11% | N/A — slower, don't use |

- MTP helps dense models, hurts MoE (routing overhead > speculation savings)
- 9B + MTP is safe for all workloads including tool calling
- 27B + MTP: use for chat/creative (70 tok/s), avoid for complex agentic (T08 regresses)
- MoE FP8 env vars: `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency`

### TP=2 Tuning (122B, 35B-tp2)

NCCL env vars for PCIe (no NVLink): `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8`

**Tested results:**
- **FIX: `NCCL_P2P_DISABLE=1` enables stable CUDA graphs on TP=2 Blackwell PCIe**
- 122B INT4: 23→**122 tok/s** (5.3x, stable over 10 runs)
- 35B MoE: 22→**205 tok/s** (9.3x, stable over 10 runs)
- Root cause: ACS enabled on PCIe bridges corrupts P2P during CUDA graph replay
- Disabling P2P forces shared memory transport — slight overhead but fully stable
- TTFT: 3077ms→29ms with prefix caching after warmup
- Power draw: only 88-96W per card at 600W limit — MoE is not power-bound
- 35B TP=2: prefix caching fixed 1.8s TTFT → 0.5s (**-70%**), wall tok/s +25%
- `VLLM_USE_FLASHINFER_MOE_FP8` crashes on 122B FP8 (unsupported quant scheme) — don't use
- **Previous finding that TP=2 needs enforce-eager was WRONG** — `NCCL_P2P_DISABLE=1` fixes it

## Running Evals

```bash
cd evals

# Profile runs (recommended)
./run.sh profile --name quick --model local    # ~15 min smoke test, 1 trial
./run.sh profile --name full --model local     # ~60-90 min comprehensive, 3 trials

# Individual runs
./run.sh claw --model local --tasks T02,T04,T06,T08 --port-offset 200
./run.sh custom --suite coding --model local --trials 1
./run.sh custom --suite reasoning --model local --trials 1
./run.sh function-call --model local --all-suites
```

### Eval Suites

| Suite | Tests | What it measures |
|-------|:-----:|-----------------|
| **claw-eval** | 52 EN | Agentic tool use (email, calendar, CRM, ops, finance) |
| **coding** | 10 | Generation (5) + analysis/review/security (5) |
| **instruction_following** | 5 | Constraint adherence, format compliance |
| **reasoning** | 5 | Math, logic puzzles, deduction, pattern recognition |
| **structured_output** | 5 | JSON, YAML, SQL, markdown tables, log parsing |
| **summarization** | 5 | Compression, action extraction, TL;DR |
| **safety** | 5 | Refusal, jailbreak resistance, PII, security review |
| **creative_writing** | 5 | Prose, narrative, character voice |
| **roleplay** | 5 | RPG GM quality, world building |
| **svg_generation** | 5 | SVG validity, accuracy, animation |
| **research** | 4 | Synthesis, conflicting sources, hallucination |
| **function_call** | 8 | Basic (5) + edge cases (3) |

### Eval Profiles

- **quick** — 6 claw tasks + 6 custom suites + FC, 1 trial (~15 min)
- **full** — 20 claw tasks + 10 custom suites + FC, 3 trials pass^3 (~60-90 min)

## Model Inventory (`/mnt/models`)

| Model | Size | tok/s | +MTP | Quick Score | Role |
|-------|------|:-----:|:----:|:-----------:|------|
| **Qwen 35B MoE FP8** | 35GB | **180** | — | — | **Speed king, 262K ctx, single GPU (Qwen official)** |
| **Qwen 9B FP8** | 19GB* | **141** | — | — | On-the-fly FP8 (+53% vs bf16) |
| **Qwen 4B FP8** | 8.8GB* | 141 | — | — | On-the-fly FP8 edge |
| **Qwen 27B INT4** | 29GB | 53 | **70** | **86/103** | Daily driver (MTP for chat, baseline for agents) |
| **Qwen 122B FP8** | 119GB | **112** | — | — | Quality ceiling FP8 TP=2 (Qwen official) |
| **Qwen 122B INT4** | 74GB | **122** | — | **89/103** | Quality ceiling INT4 TP=2 (faster on PCIe) |
| **Qwen 27B FP8 TP=2** | 29GB | **70** | — | — | 131K ctx, TP=2 (Qwen official) |
| **Qwen 9B BF16** | 19GB | 92 | **112** | 72/103 | Fine-tune base (cold storage) |
| **Qwen 4B INT4** | 3.8GB | 297 | — | 56/103 | Edge deploy (fastest absolute) |

\* On-the-fly FP8 (`--quantization fp8`) loads bf16 weights from disk, quantizes during model load. No separate quant files needed.
| Cydonia 24B | 44GB | — | — | — | Creative/roleplay (holding) |
| Llama 70B AWQ | 38GB | 38 | — | — | Creative/roleplay (holding) |
| Llama 8B AWQ | 5GB | — | — | 50/102 | Eval baseline (poor) |
| Qwen 2B BF16 | 4GB | 307 | — | — | Training (5/5 reasoning!) |
| Qwen 0.8B BF16 | 1.5GB | 547 | — | — | Training |

Base models (0.8B, 2B, 4B) also downloaded for pretraining. FP8 quants in `/mnt/models/quantized/` and on [HuggingFace protoLabsAI](https://huggingface.co/protoLabsAI).

Cold storage (`/mnt/data/models-cold/`): FLUX.2-klein 9B+base (100GB), Z-Image+Turbo (51GB), Voxtral-Mini-4B (17GB), OCR models (11.4GB).

## Blackwell GPU Constraints

- CUDA graphs work on single GPU — don't use `--enforce-eager` (37-470% speedup)
- TP=2 needs `--enforce-eager` (memory corruption under sustained load)
- `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink)
- No xformers / Flash Attention — use PyTorch native SDPA
- FlashInfer backend crashes — don't use `--attention-backend flashinfer`
- INT4 safe on dense models, unstable on MoE (use BF16 for MoE)
- Capability cliff at 4B→2B: sub-4B models can't do agentic tool use

## Secrets

All secrets in Infisical at `secrets.proto-labs.ai`. Never commit secrets. Gateway `start.sh` injects at runtime via Machine Identity.

## Storage

- `/mnt/models` — frequently-accessed model weights only (1TB NVMe, 420GB free)
- `/mnt/data` — datasets, checkpoints, outputs, cold model storage (2TB NVMe)
- `/mnt/data/models-cold/` — FLUX, Z-Image, Voxtral, OCR models (moved off fast drive)
- `/mnt/scratch` — logs, caches, docker volumes (disposable)
