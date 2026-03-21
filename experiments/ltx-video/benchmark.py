#!/usr/bin/env python3
"""
LTX-2.3 Distilled Pipeline Benchmark
Mirrors the proven Gradio demo pattern exactly.
Run: CUDA_VISIBLE_DEVICES=1 python -u benchmark.py
"""

import os
import sys
import gc
import time
import json
from pathlib import Path
from datetime import datetime

# Must be set before any torch import
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LTX_REPO_DIR = os.path.expanduser("~/dev/LTX-2")
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Blackwell: no xformers, use PyTorch native SDPA
from ltx_core.model.transformer import attention as _attn_mod
_attn_mod.memory_efficient_attention = None

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video

# Config
LTX_SNAP = "/mnt/models/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/cd784f3198e9ec3efec60b66a0fd78aafe413a86"
GEMMA_ROOT = "/mnt/models/gemma-3-12b"
OUTPUT_DIR = Path("/mnt/data/comfyui/output/benchmarks")
FRAME_RATE = 24.0
SEED = 42
WIDTH = 1536
HEIGHT = 1024

PROMPT = (
    "A cinematic live-action shot of a woman in a long dark coat walking alone "
    "through a rain-soaked city street at night. Neon reflections shimmer across "
    "the wet pavement, steam rises from street vents, and headlights pass in the "
    "far background. The camera starts in a medium frontal shot, slowly dollies "
    "backward as she walks toward lens, then gently arcs to her left side as she "
    "turns her head and looks into a glowing storefront window. Her coat sways "
    "naturally, raindrops bead on the fabric, and the atmosphere feels moody, cool, "
    "and realistic. Ambient audio includes soft rain, distant traffic, and a faint "
    "subway rumble."
)

DURATIONS = [5, 10, 15, 30, 45, 60]


def get_gpu_mem():
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**2
    total_mb = total / 1024**2
    return used, total_mb


@torch.inference_mode()
def run_single(pipeline, duration_sec):
    """Run a single generation — mirrors Gradio generate_video() exactly."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    num_frames = int(duration_sec * FRAME_RATE) + 1
    num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

    mem_before, mem_total = get_gpu_mem()
    print(f"\n{'='*60}")
    print(f"  {duration_sec}s | {num_frames} frames | {WIDTH}x{HEIGHT}")
    print(f"  VRAM before: {mem_before:.0f} MiB / {mem_total:.0f} MiB")
    print(f"{'='*60}")

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    t_start = time.perf_counter()

    video, audio = pipeline(
        prompt=PROMPT,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        frame_rate=FRAME_RATE,
        images=[],
        tiling_config=tiling_config,
        enhance_prompt=False,
    )

    t_gen = time.perf_counter() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    mem_after, _ = get_gpu_mem()

    # Encode
    output_path = OUTPUT_DIR / f"bench_{duration_sec}s_{SEED}.mp4"
    t_enc_start = time.perf_counter()
    encode_video(
        video=video,
        fps=FRAME_RATE,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )
    t_enc = time.perf_counter() - t_enc_start

    # Free generation tensors
    del video, audio
    gc.collect()
    torch.cuda.empty_cache()

    file_size_mb = output_path.stat().st_size / 1024**2

    result = {
        "duration_sec": duration_sec,
        "num_frames": num_frames,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "fps": FRAME_RATE,
        "seed": SEED,
        "gen_time_sec": round(t_gen, 2),
        "encode_time_sec": round(t_enc, 2),
        "total_time_sec": round(t_gen + t_enc, 2),
        "peak_vram_mb": round(peak_mem),
        "peak_vram_gb": round(peak_mem / 1024, 1),
        "vram_before_mb": round(mem_before),
        "vram_after_mb": round(mem_after),
        "file_size_mb": round(file_size_mb, 1),
        "output_path": str(output_path),
        "sec_per_sec_video": round(t_gen / duration_sec, 2),
    }

    print(f"  Gen:   {result['gen_time_sec']}s ({result['sec_per_sec_video']}s per 1s video)")
    print(f"  Enc:   {result['encode_time_sec']}s")
    print(f"  Peak:  {result['peak_vram_gb']} GB")
    print(f"  File:  {result['file_size_mb']} MB")

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 60)
    print("  LTX-2.3 Distilled Benchmark")
    print(f"  GPU: {gpu_name}")
    print(f"  VRAM: {gpu_vram:.1f} GB")
    print(f"  Res: {WIDTH}x{HEIGHT} @ {FRAME_RATE}fps")
    print(f"  Quant: fp8-cast")
    print(f"  Durations: {DURATIONS}")
    print("=" * 60)

    # Init pipeline (lazy — no VRAM used yet)
    print("\nInitializing pipeline...")
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=os.path.join(LTX_SNAP, "ltx-2.3-22b-distilled.safetensors"),
        spatial_upsampler_path=os.path.join(LTX_SNAP, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        gemma_root=GEMMA_ROOT,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )
    print("Pipeline ready.")

    # Run benchmarks
    results = []
    for dur in DURATIONS:
        try:
            result = run_single(pipeline, dur)
            results.append(result)
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  OOM at {dur}s: {e}")
            results.append({"duration_sec": dur, "error": "OOM", "detail": str(e)[:100]})
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n  FAILED at {dur}s: {e}")
            results.append({"duration_sec": dur, "error": str(type(e).__name__), "detail": str(e)[:100]})
            gc.collect()
            torch.cuda.empty_cache()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "gpu": gpu_name,
                "vram_gb": round(gpu_vram, 1),
                "resolution": f"{WIDTH}x{HEIGHT}",
                "fps": FRAME_RATE,
                "quantization": "fp8-cast",
                "model": "ltx-2.3-22b-distilled",
                "prompt": PROMPT,
                "seed": SEED,
                "timestamp": timestamp,
            },
            "results": results,
        }, f, indent=2)

    # Summary
    print(f"\n\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"{'Dur':>6} {'Frames':>7} {'Gen':>8} {'Enc':>7} {'Peak':>8} {'Size':>7} {'s/s':>6}")
    print(f"{'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        if "error" in r:
            print(f"{r['duration_sec']:>5}s {'':>7} {r['error']}")
        else:
            print(f"{r['duration_sec']:>5}s {r['num_frames']:>7} {r['gen_time_sec']:>7}s {r['encode_time_sec']:>6}s {r['peak_vram_gb']:>6} GB {r['file_size_mb']:>5} MB {r['sec_per_sec_video']:>5}s")

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
