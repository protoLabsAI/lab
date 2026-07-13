# Quantizing LTX-2.3 22B to NVFP4 on Blackwell (and the cu130 trap everyone hits)

Lightricks ships an fp4 checkpoint for **LTX-2 19B**, but the newer, better **LTX-2.3 22B is bf16 only**. If you
have a Blackwell card (RTX PRO 6000, RTX 50-series) that means leaving the 4-bit TensorCores idle on your best
video model. So we quantized it ourselves — and reverse-engineered Lightricks' exact mixed-precision policy in
the process.

## The result

A 22.9 GB NVFP4 checkpoint (down from 46.1 GB), running the DiT at **1.57× the speed of bf16** and **−38% peak
VRAM**, with the distilled decode path **visually indistinguishable** from bf16.

| 960×544, 8-step distilled | bf16 | NVFP4 | Δ |
|---|---|---|---|
| Disk | 46.1 GB | 22.9 GB | 2.0× |
| DiT step | 2.85 s/it | 1.82 s/it | 1.57× |
| Peak VRAM | ~60 GB | ~37 GB | −38% |
| Cold load + first clip | 39.2 s | 19.0 s | 2.1× |

(Reproduced across 2 runs each. Speedup is on the DiT denoising loop; the bf16 VAE decode is unchanged, so
whole-pipeline gain is smaller on very short clips.)

## The trap: cu130 or bust

The single biggest source of "NVFP4 is slower than fp8" confusion in the LTX community is the CUDA version.
NVFP4 only hits the fast path on **torch built with CUDA 13**. On cu128 it silently falls back to a slow path
and runs **~2× slower than fp8** — ComfyUI even warns you (`You need pytorch with cu130 or higher to use
optimized CUDA operations`) but it's easy to miss. Same sm120/cu130 story we hit on the vLLM side. One
force-reinstall fixes it:

```bash
pip install --force-reinstall --no-cache-dir torch==2.11.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

## The method: mirror Lightricks, don't guess

There's no turnkey "quantize an LTX checkpoint" node. But `comfy_kitchen` (shipped with ComfyUI-LTXVideo)
exposes the exact primitive the official fp4 was built with — `TensorCoreNVFP4Layout.quantize`. The trick is
matching two things so ComfyUI's loader recognizes the result:

1. **The tensor format.** Each fp4 weight is three tensors: `W` (uint8 packed), `W_scale` (fp8-e4m3 per-block),
   `W_scale_2` (fp32 per-tensor). Using `comfy_kitchen`'s own serializer guarantees this.
2. **The `_quantization_metadata` header.** This is the actual load trigger. Without it the loader treats the
   packed uint8 as a regular weight and dies on a shape mismatch (`[4096,2048]` vs `[4096,4096]`). It's a JSON
   map of `{module_path: {"format": "nvfp4"}}` in the safetensors header.

And **which** layers to quantize? We diffed the shipped 19B fp4 against its bf16 and recovered the policy: fp4
the transformer-block Linears, but **keep block 0, the last 5 blocks, all gating projections, and the entire
VAE/vocoder in bf16**. Classic first/last-layer precision retention — the bf16 blocks absorb the ~9%
per-tensor error of the 4-bit middle. We applied the identical policy to the 22B.

## The honest caveat: distilled decode only

The 2.3 workflow has two decode branches. On the fp4 model, the **distilled decode is clean**, but the **full
decode shows mild added artifacting**. Not surprising — the DiT is where the 4-bit lives, and the full VAE
decoder is more sensitive to it. Default to distilled decode and it's a clean win.

## Reproduce

The converter is ~100 lines, runs in ~40 s on one Blackwell card, and is in
[`protoLabsAI/lab`](https://github.com/protoLabsAI) under `experiments/ltx2-nvfp4/`. The checkpoint is on
[HuggingFace](https://huggingface.co/protoLabsAI).

---

*Part of the protoLabs quant + serving lab — parity-verified low-bit models and the Blackwell serving findings
that make them run. LTX-2.3 · NVFP4 · sm120 · cu130.*
