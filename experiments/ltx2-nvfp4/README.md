---
license: other
license_name: ltx-2-community-license
license_link: LICENSE
base_model: Lightricks/LTX-2.3
tags:
  - text-to-video
  - image-to-video
  - video-generation
  - nvfp4
  - fp4
  - quantization
  - blackwell
  - comfyui
  - ltx-video
library_name: comfyui
pipeline_tag: text-to-video
---

# LTX-2.3-22B-distilled — NVFP4 (mixed precision)

A **native NVFP4** (4-bit) quantization of [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3)
`22b-distilled-1.1`, for **NVIDIA Blackwell** (RTX PRO 6000 / RTX 50-series, sm120) via ComfyUI.

Lightricks ships an fp4 checkpoint for **LTX-2 19B** but the newer **LTX-2.3 22B is bf16-only**. This fills
that gap: the 22B distilled model at ~half the disk and VRAM, running on Blackwell's native FP4 TensorCores.

## Results (measured, honest)

Single-clip, `960×544`, 8-step distilled sampler, distilled decode, RTX PRO 6000 (96 GB), vLLM-adjacent
ComfyUI stack on **torch cu130**. Speed/VRAM reproduced across 2 runs each; quality eyeballed across 4 prompts.

| Metric | bf16 (upstream) | **NVFP4 (this)** | Δ |
|---|---|---|---|
| Disk size | 46.1 GB | **22.9 GB** | **2.0× smaller** |
| DiT step speed | 2.85 s/it | **1.82 s/it** | **1.57× faster** |
| Peak runtime VRAM | ~60 GB | **~37 GB** | **−38%** |
| Cold load + first clip | 39.2 s | **19.0 s** | **2.1× faster** |
| Distilled-decode quality | baseline | **visually unchanged** | — |

Higher resolution holds the win (fp4 `1280×704` ≈ 3.85 s/it, scaling with token count as expected).

## ⚠️ You need torch built with CUDA 13 (cu130)

This is the single most important thing. NVFP4 only hits the fast TensorCore path on **cu130**. On **cu128,
fp4 silently falls back and runs ~2× *slower* than fp8** (ComfyUI logs `WARNING: You need pytorch with cu130 or
higher to use optimized CUDA operations`). This is the field's #1 LTX-fp4 confusion. If your fp4 is slower than
fp8, this is why.

```bash
pip install --force-reinstall --no-cache-dir torch==2.11.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

## ⚠️ Use the distilled decode path

The LTX-2.3 single-stage workflow ships **two decode branches** (a fast "distilled" decode and a "full"
decode). On this quant, the **distilled decode is visually indistinguishable from bf16**, while the **full
decode shows mild added artifacting**. Default to the distilled decode. (The DiT is where the fp4 lives; the
VAE decoders are untouched bf16 — the full decoder is just more sensitive to the small DiT-linear error.)

## Usage (ComfyUI)

1. cu130 torch (above) + [`ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo)
   (provides the `comfy_kitchen` NVFP4 kernels: `scaled_mm_nvfp4`, `quantize_nvfp4`).
2. Drop `ltx-2.3-22b-distilled-1.1-fp4.safetensors` into `ComfyUI/models/checkpoints/`.
3. Load the `2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full` example workflow, point the
   `CheckpointLoaderSimple` at this file, keep the **distilled** decode/save branch.
4. Text encoder: any Gemma-3-12B LTX encoder (`gemma_3_12B_it_fp8_scaled.safetensors` works well).

The checkpoint carries a standard `_quantization_metadata` header, so ComfyUI's LTX loader auto-detects the
NVFP4 layers — no flags, no config edits.

## What's quantized (mixed precision)

Replicates Lightricks' own 19B-fp4 layer policy exactly (reverse-engineered from their shipped checkpoint):

- **NVFP4 (4-bit):** the 2-D Linear weights in `transformer_blocks` **1–42** (attention q/k/v/out + feed-forward,
  video + audio + cross-modal). 1,176 weights.
- **Kept bf16 (precision-sensitive):** transformer block **0** and the **last 5 blocks (43–47)**, all
  `to_gate_logits` gates, the patchify / proj_out / adaln / caption + embeddings connectors, and the entire
  VAE / audio-VAE / vocoder.

Each fp4 weight is stored as `W` (uint8 packed) + `W_scale` (fp8-e4m3 per-block) + `W_scale_2` (fp32
per-tensor) — byte-identical to the shipped 19B fp4 format.

## Reproduce it

The full converter is `quantize.py` (in this repo / `protoLabsAI/lab`). It streams the bf16 checkpoint
tensor-by-tensor through `comfy_kitchen`'s own `TensorCoreNVFP4Layout.quantize`, so the output format is
guaranteed loader-compatible. ~40 s on one Blackwell card.

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py \
  --in  ltx-2.3-22b-distilled-1.1.safetensors \
  --out ltx-2.3-22b-distilled-1.1-fp4.safetensors
```

## Limitations & honest notes

- Numbers are **single-clip** on one specific Blackwell + cu130 ComfyUI config; your mileage varies with
  resolution, length, sampler, and card.
- Speedup is on the **DiT denoising** loop. End-to-end wall-clock also includes VAE decode (bf16, unchanged)
  and text encoding, so the *whole-pipeline* ratio is smaller than 1.57× at short clip lengths.
- **Full-decode** path: mild artifacting (see above). Distilled decode only.
- FP4 is 4-bit — expect a small per-tensor error (~9% relative L1 on a quantized layer), which the first/last
  bf16 blocks are there to absorb. If you see quality loss, widen the bf16-kept block set and re-quantize.

## Attribution & license

Derivative of [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3), released under the **LTX-2
Community License** (see `LICENSE`). This quant is redistributed under the same license, including its
Acceptable Use Policy (Attachment A) and paragraph 4 use restrictions, which pass through to you. High-revenue
commercial entities require a separate Commercial Use Agreement from Lightricks for *use* — see the license.

Quantized by [protoLabsAI](https://huggingface.co/protoLabsAI). Method + measurements:
`protoLabsAI/lab` → `experiments/ltx2-nvfp4/`.
