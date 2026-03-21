# LTX-2.3 Video Generation

Experiments with LTX-2.3 on 2x RTX PRO 6000 Blackwell (192 GB VRAM).

## Models

All on `/mnt/models/huggingface/`:

- `Lightricks/LTX-2.3` — 22B video gen (dev + distilled + upsamplers)
- `google/gemma-3-12b-it-qat-q4_0-unquantized` — text encoder at `/mnt/models/gemma-3-12b/`

## Experiments

### `gradio-demo.py`

Gradio UI for text/image-to-video using the distilled pipeline with fp8 quantization. Port 7860.

## Generation Settings

| Workflow | Model | Steps | CFG | Use Case |
|----------|-------|-------|-----|----------|
| Fast draft | Distilled | 8+4 | 1.0 | Iteration, previews |
| Quality render | Dev | 40 | 3.0-3.5 | Final output |
| Two-stage HQ | Dev + upsampler | 15 | 3.0 | Best quality, res_2s sampler |

## VRAM Benchmarks (distilled, fp8, single GPU)

| Duration | Peak VRAM | Headroom (96 GB) |
|----------|-----------|------------------|
| 5s | ~28 GB | 68 GB |
| 10s | ~32 GB | 64 GB |
| 30s | ~52 GB | 44 GB |
| 45s | ~68 GB | 28 GB |
| 60s | ~85 GB | 11 GB |

## Blackwell Notes

- xformers does NOT work (compute 12.0 too new) — use PyTorch native SDPA
- Flash Attention 2/3 unsupported on Blackwell; FA4 forward-only, awaiting CUTLASS 4.4
- torch.compile not yet integrated into LTX-2.3 (watch GitHub Issue #421)
- FP4 on Blackwell proven for FLUX (6.3x speedup) but not yet available for LTX-2.3
- Use bf16 for max quality when VRAM allows; fp8-cast for speed
- Spatial upscaler x2 v1.1 fixes endscreen artifacts from v1.0
- Temporal upscaler x2 can double frame count post-generation

## Prompting

- 100-200 words, chronological, present tense
- Describe: shot setup, action, environment, camera moves, lighting, audio
- Match prompt length to video duration (short prompt + long video = rushed action)
- Use `enhance_prompt=True` for automatic expansion
