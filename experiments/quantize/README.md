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
| `nvfp4` | llm-compressor | 512 samples | ~72% | dense: −1.5–2.5 MMLU-Pro | ~30 min | ~24 GB |
| `nvfp4a16` | llm-compressor | None | ~72% | ≈nvfp4 | ~15 min | ~24 GB |

**`nvfp4`/`nvfp4a16` require quant-env** (`~/dev/quant-env/`, llm-compressor ≥0.10 with NVFP4 presets), not vllm-env.

## NVFP4 (Blackwell-native 4-bit)

E2M1 values, 16-element blocks with FP8 E4M3 fractional scales + per-tensor FP32
scale. Dequant happens inside the sm120 tensor core (`mma.sync...e2m1.e2m1`).
`nvfp4` = W4A4 (the hardware path, calibrated); `nvfp4a16` = weight-only,
data-free fallback if the W4A4 serve path fights back.

```bash
source ~/dev/quant-env/bin/activate
python quantize.py --model deepreinforce-ai/Ornith-1.0-9B --method nvfp4
```

Recipe details for the Qwen3.5 hybrid family (Ornith):

- **DeltaNet (`linear_attn`) stays bf16** — low-precision activations corrupt it
  (same class as the W8A8-FP8 finding on this arch). Vision tower and `lm_head`
  also excluded; excluding `lm_head` sidesteps the vLLM 0.22.1 quantized-lm_head
  loader gap that blocked the ModelOpt Qwen3.6-27B-NVFP4 checkpoint.
- **MTP composes cleanly**: the draft head is a bf16 sidecar
  (`protoLabsAI/Ornith-1.0-9B-MTP`, `model-mtp.safetensors`) loaded via
  `--speculative-config` — never part of the quantized checkpoint. Spec decode
  verifies against the quantized target, so MTP+NVFP4 output ≡ NVFP4-only
  output; only the acceptance rate can drift (head was distilled on bf16
  Ornith logits — re-measure accept% and re-distill only if it craters).

Serving gates on sm120 (vLLM 0.22.1):

- FlashInfer NVFP4 cutlass + fused-MoE GEMMs need the local kernel build —
  `FLASHINFER-SM120-RECIPE.md` (CUDA_HOME at bundled CUDA-13 + dev symlinks).
- `VLLM_USE_TRITON_FP8_GEMM=1` (the FlashInfer FP8 matmul silently deadlocks
  on Blackwell) + the usual `VLLM_USE_FLASHINFER_SAMPLER=0`.
- If W4A4 won't serve, `nvfp4a16` weight-only loads through the
  compressed-tensors marlin-style path with none of the cutlass dependencies.

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
