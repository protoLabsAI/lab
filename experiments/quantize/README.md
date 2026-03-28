# Model Quantization Pipeline

Quantize HuggingFace models for vLLM serving, benchmark speed, and quality-gate with evals.

## Quick Start

```bash
# Activate vllm-env (has torch, transformers, vllm)
source ~/dev/vllm-env/bin/activate

# Install quantization tools (one-time)
pip install llmcompressor autoawq

# FP8 — fastest, no calibration, ~50% size, ~99% quality
python quantize.py --model Qwen/Qwen3.5-9B --method fp8

# GPTQ INT4 — calibrated, ~75% size, ~95% quality
python quantize.py --model Qwen/Qwen3.5-9B --method gptq

# AWQ INT4 — calibrated, fast, ~75% size, ~95% quality
python quantize.py --model Qwen/Qwen3.5-9B --method awq
```

## Full Pipeline (Quantize → Serve → Speed Test → Eval)

```bash
# FP8 with speed test
bash benchmark.sh Qwen/Qwen3.5-9B fp8

# GPTQ with speed test + eval
bash benchmark.sh Qwen/Qwen3.5-9B gptq --eval
```

## Methods

| Method | Tool | Calibration | Size | Quality | Time (9B) | VRAM (9B) |
|--------|------|:-----------:|:----:|:-------:|:---------:|:---------:|
| `fp8` | llm-compressor | None | ~50% | ~99% | ~10 min | ~24 GB |
| `gptq` | llm-compressor | 512 samples | ~75% | ~95% | ~1-2 hrs | ~24 GB |
| `awq` | AutoAWQ | 128 samples | ~75% | ~95% | ~15 min | ~24 GB |

## Custom Calibration Data

For domain-tuned quantization (agentic/tool-calling), provide your own calibration dataset:

```bash
python quantize.py --model Qwen/Qwen3.5-9B --method gptq \
    --calib-dataset training/data/my_agentic_data.jsonl
```

JSONL format: one JSON object per line with a `"text"` field.

## Output

Quantized models are saved to `/mnt/models/quantized/<model>-<method>/` and are directly loadable by vLLM:

```bash
vllm serve /mnt/models/quantized/Qwen3.5-9B-FP8 --host 0.0.0.0 --port 8000
```

## Targets

| Model | bf16 Size | FP8 Size | INT4 Size | Use Case |
|-------|:---------:|:--------:|:---------:|----------|
| Qwen3.5-9B | 19 GB | ~10 GB | ~5 GB | Fine-tune base, ava A6000 deploy |
| Qwen3.5-27B | 52 GB | ~26 GB | ~14 GB | Single-GPU daily driver |
| Qwen3.5-35B MoE | 67 GB | ~34 GB | — | Speed king (INT4 unstable on MoE) |

## Push to Hub

```bash
python quantize.py --model Qwen/Qwen3.5-9B --method fp8 --push-to-hub
# → ArtificialCitizens/Qwen3.5-9B-FP8
```
