# CLAUDE.md — protoLabs AI Lab

Monorepo for model evaluation, training, inference infrastructure, and ML experiments.

## Structure

- `packages/lab-core/` — Shared Pydantic models, GPU utils (strict, tested, publishable)
- `evals/` — LLM eval suite: claw-eval, function-call, RAG, creative, coding (strict, tested)
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
bash models/vllm-swap.sh qwen-27b-int4    # daily driver (44 tok/s, 3/4 pass^3)
bash models/vllm-swap.sh qwen-35b          # speed king (170 tok/s MoE)
bash models/vllm-swap.sh qwen-122b-int4-1gpu  # highest quality (30 tok/s)
```

## Running Evals

```bash
cd evals
./run.sh claw --model local --tasks T02,T04,T06,T08 --port-offset 200
./run.sh custom --suite creative_writing --model local --trials 1
./run.sh function-call --model local --all-suites
```

## Blackwell GPU Constraints

- CUDA graphs work on single GPU — don't use `--enforce-eager` (37-470% speedup)
- TP=2 needs `--enforce-eager` (memory corruption under sustained load)
- `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink)
- No xformers / Flash Attention — use PyTorch native SDPA
- FlashInfer backend crashes — don't use `--attention-backend flashinfer`
- INT4 safe on dense models, unstable on MoE (use BF16 for MoE)

## Secrets

All secrets in Infisical at `secrets.proto-labs.ai`. Never commit secrets. Gateway `start.sh` injects at runtime via Machine Identity.

## Storage

- `/mnt/models` — model weights only (1TB NVMe)
- `/mnt/data` — datasets, checkpoints, outputs (2TB NVMe)
- `/mnt/scratch` — logs, caches, docker volumes (disposable)
