# CLAUDE.md — protoLabs AI Lab

Monorepo for model evaluation, training, inference infrastructure, and ML experiments.

## Structure

- `packages/lab-core/` — Shared Pydantic models, GPU utils (strict, tested, publishable)
- `evals/` — LLM eval suite: claw-eval, custom suites, function-call, RAG (strict, tested)
- `models/` — Model inventory, vllm-swap.sh, benchmarks
- `training/` — Fine-tuning workspace (LLaMA-Factory configs, datasets)
- `experiments/` — Loose ML experiment scripts (ltx-video, flux2)
- `infra/` — Gateway (LiteLLM + Langfuse), vLLM systemd, Prometheus configs

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
bash models/vllm-swap.sh qwen-27b-int4     # daily driver (44 tok/s, 3/4 pass^3)
bash models/vllm-swap.sh qwen-27b-int4-opt # daily driver + P1+P2 optimizations
bash models/vllm-swap.sh qwen-35b          # speed king (170 tok/s MoE)
bash models/vllm-swap.sh qwen-122b-int4-1gpu  # highest quality (30 tok/s)
bash models/vllm-swap.sh qwen-9b           # fine-tune base (92 tok/s)
bash models/vllm-swap.sh qwen-4b-int4      # edge deploy (294 tok/s)
bash models/vllm-swap.sh qwen-4b-int4-opt  # edge deploy + P1+P2 optimizations
bash models/vllm-swap.sh qwen-4b           # edge deploy bf16 (155 tok/s)
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
- 122B: NCCL tuning has **zero impact** (18.5→18.4 tok/s, within noise)
- 35B TP=2: prefix caching fixed 1.8s TTFT → 0.5s (**-70%**), wall tok/s +25%
- `VLLM_USE_FLASHINFER_MOE_FP8` crashes on 122B FP8 (unsupported quant scheme) — don't use
- Inference at 300W draws only ~140W per card — MoE is not power-bound

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

| Model | Size | tok/s | +MTP | Claw pass^3 | Role |
|-------|------|:-----:|:----:|:-----------:|------|
| **Qwen 27B INT4** | 29GB | 53 | **70** | 3/4 | Daily driver, all-rounder |
| **Qwen 35B MoE BF16** | 67GB | 171 | — | 3/4 | Speed king (single GPU, 220 TP=2) |
| **Qwen 122B FP8** | 119GB | 18.5 | — | 3/4 | Quality ceiling (TP=2 only) |
| **Qwen 122B INT4** | 74GB | ~30 | — | 3/4 | Quality ceiling (TP=2, smaller) |
| **Qwen 9B BF16** | 19GB | 92 | **112** | 2/4 | Fine-tune base (best coding at 9B) |
| **Qwen 4B INT4** | 3GB | 297 | — | 2-3/4 | Edge deploy speed demon |
| **Qwen 4B BF16** | 8GB | 155 | — | 3/4 | Edge deploy, LoRA base |
| Cydonia 24B | 44GB | — | — | 0/4 | Creative/roleplay only |
| Llama 70B AWQ | 38GB | 38 | — | 1/4 | Creative/roleplay |
| Qwen 2B BF16 | 4GB | 307 | — | 0/4 | Training experiments |
| Qwen 0.8B BF16 | 1.5GB | 547 | — | 0/4 | Training experiments |

Base models (0.8B, 2B, 4B) also downloaded for pretraining. 126GB free on `/mnt/models`.

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

- `/mnt/models` — model weights only (1TB NVMe)
- `/mnt/data` — datasets, checkpoints, outputs (2TB NVMe)
- `/mnt/scratch` — logs, caches, docker volumes (disposable)
