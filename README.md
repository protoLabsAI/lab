# protoLabs AI Lab

Monorepo for model evaluation, training, inference infrastructure, and ML experiments on prosumer GPUs.

## Hardware

- 2x NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM each, 192 GB total)
- CUDA 12.8, Driver 595.x
- CUDA graphs work on single GPU (37-470% speedup depending on model)

## Projects

| Package | Purpose | Strictness |
|---------|---------|------------|
| **lab-core** | Shared Pydantic models, GPU utils, path constants | Strict + tests |
| **evals** | LLM evaluation suite (claw-eval, function-call, RAG, creative, coding) | Strict + tests |
| **models** | Model inventory, vllm-swap configs, benchmarks | Mixed |
| **training** | Fine-tuning workspace (LLaMA-Factory, TRL) | Loose |
| **experiments** | ML experiments (video gen, image gen, demos) | Loose |

## Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| **Gateway** | `infra/gateway/` | LiteLLM proxy — unified API for 20+ LLM providers |
| **Langfuse** | `infra/gateway/docker-compose.yml` | LLM observability (traces, scores, experiments) |
| **vLLM** | `infra/vllm/` | Local LLM inference (systemd service) |
| **Prometheus** | `infra/prometheus/` | Metrics collection + alert rules |

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all projects
uv sync

# Run evals
uv run proto-eval claw --model local --tasks T02,T04,T06,T08

# Show model inventory
uv run models --gpu single

# Swap vLLM model
bash models/vllm-swap.sh qwen-27b-int4

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Model Leaderboard (Claw-Eval Agent Tasks)

| Rank | Model | tok/s | pass^3 | Avg Score | Config |
|------|-------|:-----:|:------:|:---------:|--------|
| 1 | Qwen 35B MoE BF16 TP=2 | 170 | 3/4 | 0.80 | Both GPUs, 250K ctx |
| 2 | Qwen 27B INT4 | 44 | 3/4 | 0.79 | Single GPU, 160K ctx |
| 3 | Qwen 122B INT4 1GPU | ~30 | 3/4 | 0.78 | enforce-eager, 64K |
| 4 | OmniCoder 9B | 92 | 2/4 | 0.76 | Single GPU, 262K ctx |
| 5 | Llama 70B AWQ | 38 | 1/4 | 0.65 | Creative writing only |

Cloud comparison: GLM 5 Turbo (0.85), Sonnet 4.6 (0.85), Opus 4.6 (0.84) are the top cloud models.

## Architecture

```
lab/
├── packages/lab-core/     Pydantic models, GPU utils (publishable)
├── evals/                 Eval suite (publishable)
├── models/                Model configs + inventory
├── training/              Fine-tuning workspace
├── experiments/           ML experiments (loose scripts)
└── infra/
    ├── gateway/           LiteLLM + Langfuse docker stack
    ├── vllm/              systemd service configs
    └── prometheus/        Metrics + alert rules
```

## Key Findings

- **CUDA graphs on Blackwell**: 37-470% speedup. MoE models benefit most (3B active params → 170 tok/s).
- **INT4 on dense models**: No quality loss vs BF16. Use GPTQ-Int4 for dense, BF16 for MoE.
- **MoE INT4 instability**: Quantization corrupts expert routing. Keep MoE at BF16.
- **Power draw**: Inference uses 300-340W per GPU regardless of power limit.

## Secrets

All secrets managed by Infisical. Zero secrets in this repo. Gateway `start.sh` authenticates via Machine Identity and injects env vars at runtime.
