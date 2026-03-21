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
bash models/vllm-swap.sh qwen-35b          # speed king (170 tok/s MoE)
bash models/vllm-swap.sh qwen-122b-int4-1gpu  # highest quality (30 tok/s)
bash models/vllm-swap.sh qwen-9b           # fine-tune base (92 tok/s)
bash models/vllm-swap.sh qwen-4b-int4      # edge deploy (294 tok/s)
bash models/vllm-swap.sh qwen-4b           # edge deploy bf16 (155 tok/s)
```

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

| Model | Size | tok/s | Claw pass^3 | Role |
|-------|------|:-----:|:-----------:|------|
| **Qwen 27B INT4** | 14GB | 44 | 3/4 | Daily driver, all-rounder |
| **Qwen 35B MoE BF16** | 67GB | 170 | 3/4 | Speed king (TP=2) |
| **Qwen 122B INT4** | 74GB | ~30 | 3/4 | Quality ceiling |
| **Qwen 9B BF16** | 19GB | 92 | 2/4 | Fine-tune base (best coding at 9B) |
| **Qwen 4B INT4** | 3GB | 294 | 2-3/4 | Edge deploy speed demon |
| **Qwen 4B BF16** | 8GB | 155 | 3/4 | Edge deploy, LoRA base |
| Qwen 2B BF16 | 4GB | 307 | 0/4 | Training experiments |
| Qwen 0.8B BF16 | 1.5GB | 547 | 0/4 | Training experiments |
| Qwen 27B BF16 | 52GB | 44 | 3/4 | Baseline (can nuke for 52GB) |
| Llama 70B AWQ | 38GB | 38 | 1/4 | Creative/roleplay |

Base models (0.8B, 2B, 4B) also downloaded for pretraining.

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
