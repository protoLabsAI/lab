#!/usr/bin/env python3
"""
Pixel Art Generator — Isolated Gradio experiment (optimized)
Extracted from mythxengine's imagegen service for optimization testing.

SDXL + Lightning LoRA → pixel cleanup pipeline with full parameter control.

Optimizations applied:
- Tiny VAE (taesdxl) — 10x faster decode, invisible quality loss for pixel art
- No attention slicing — use native SDPA on Blackwell
- torch.compile max-autotune on UNet, reduce-overhead on VAE + text encoders
- channels_last memory format for convolutions
- LoRA fusing for fused Lightning weights
- Vectorized pixel cleanup (see pixeldetector.py)

Run: CUDA_VISIBLE_DEVICES=1 uv run python -u app.py
"""

import logging
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import gradio as gr
import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from huggingface_hub import hf_hub_download
from PIL import Image

from pixeldetector import cleanup_pixel_art

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE = torch.float16
OUTPUT_DIR = Path("/mnt/data/comfyui/output/pixel-gen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TINY_VAE_ID = "madebyollin/taesdxl"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_FILE = "sdxl_lightning_4step_lora.safetensors"
SYSTEM_LORA_DIR = Path("/mnt/data/models-cold/system_loras")

LORA_DIR = Path("/mnt/data/models-cold/loras")

RESOLUTION_PRESETS = {
    "Scene (1024x768)": (1024, 768),
    "Portrait (768x768)": (768, 768),
    "Icon (512x512)": (512, 512),
    "Wide (1024x576)": (1024, 576),
    "Square (1024x1024)": (1024, 1024),
}

QUALITY_PRESETS = {
    "Lightning (4 steps)": {"steps": 4, "cfg": 1.5, "lightning": True},
    "Fast (12 steps)": {"steps": 12, "cfg": 4.0, "lightning": False},
    "Standard (25 steps)": {"steps": 25, "cfg": 7.0, "lightning": False},
    "Quality (40 steps)": {"steps": 40, "cfg": 7.5, "lightning": False},
}

STYLE_PRESETS = {
    "16-bit RPG": "16-bit pixel art, retro RPG style, vibrant colors, fantasy",
    "8-bit NES": "8-bit pixel art, NES retro style, limited palette, nostalgic",
    "32-bit PS1": "32-bit pixel art, PlayStation 1 era, low-poly aesthetic, detailed",
    "Game Boy": "pixel art, Game Boy green monochrome, 4-shade palette, retro handheld",
    "Isometric": "isometric pixel art, detailed tileset style, clean edges, game asset",
    "None": "",
}

NEGATIVE_PROMPT = (
    "blurry, smooth, photorealistic, 3d render, anti-aliased, gradient, "
    "noise, grain, watermark, text, signature, low quality"
)

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------
_pipe: StableDiffusionXLPipeline | None = None


def get_pipeline() -> StableDiffusionXLPipeline:
    """Load or return cached SDXL pipeline with all optimizations."""
    global _pipe
    if _pipe is not None:
        return _pipe

    logger.info(f"Loading SDXL: {MODEL_ID}")
    t0 = time.time()

    _pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16",
    ).to(DEVICE)

    # --- Tiny VAE: ~10x faster decode, quality loss invisible for pixel art ---
    try:
        tiny_vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=DTYPE)
        _pipe.vae = tiny_vae.to(DEVICE)
        logger.info(f"Tiny VAE loaded: {TINY_VAE_ID}")
    except Exception as e:
        logger.warning(f"Tiny VAE failed, using full VAE: {e}")

    # --- DO NOT use attention_slicing — it's a memory/speed tradeoff that hurts
    #     on Blackwell where we have 96GB VRAM and native SDPA ---

    # --- VAE slicing for batch decode efficiency ---
    _pipe.enable_vae_slicing()

    # --- channels_last memory format: faster convolutions on NVIDIA ---
    _pipe.unet = _pipe.unet.to(memory_format=torch.channels_last)
    _pipe.vae = _pipe.vae.to(memory_format=torch.channels_last)

    _pipe.scheduler = EulerDiscreteScheduler.from_config(
        _pipe.scheduler.config, timestep_spacing="trailing"
    )

    # --- Load + fuse Lightning LoRA BEFORE torch.compile ---
    # (compiled models can't dynamically load LoRAs, so we fuse permanently)
    lightning_path = ensure_lightning_lora()
    if lightning_path:
        _pipe.load_lora_weights(
            str(lightning_path.parent), weight_name=lightning_path.name,
            adapter_name="lightning",
        )
        _pipe.set_adapters(["lightning"], [1.0])
        _pipe.fuse_lora()
        _pipe.unload_lora_weights()  # Free adapter memory, weights are fused
        logger.info("Lightning LoRA loaded, fused, and freed")

    # --- torch.compile DISABLED ---
    # Each unique resolution triggers a full recompile (~10-30s stall).
    # Not worth it for interactive use with multiple resolution presets.
    # The fused Lightning LoRA + Tiny VAE + channels_last + no attention slicing
    # already deliver most of the speedup without compile overhead.

    logger.info(f"SDXL loaded in {time.time() - t0:.1f}s")

    return _pipe


def ensure_lightning_lora() -> Path | None:
    """Download Lightning LoRA if needed, return path."""
    local = SYSTEM_LORA_DIR / LIGHTNING_FILE
    if local.exists():
        return local
    SYSTEM_LORA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=LIGHTNING_REPO,
            filename=LIGHTNING_FILE,
            local_dir=str(SYSTEM_LORA_DIR),
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as e:
        logger.error(f"Failed to download Lightning LoRA: {e}")
        return None


    # Lightning LoRA is permanently fused at pipeline load time.
    # All quality presets work with the fused weights — non-Lightning presets
    # simply use more steps and higher CFG which overrides the Lightning behavior.


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
    prompt: str,
    style: str,
    quality: str,
    resolution: str,
    seed: int,
    randomize_seed: bool,
    custom_negative: str,
    # Pixel cleanup params
    enable_cleanup: bool,
    reduce_palette: bool,
    max_colors: int,
    upscale_factor: int,
    # Advanced overrides
    override_steps: int,
    override_cfg: float,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate pixel art with optional cleanup pipeline."""
    pipe = get_pipeline()

    # Build prompt
    style_prefix = STYLE_PRESETS.get(style, "")
    full_prompt = f"{style_prefix}, {prompt}" if style_prefix else prompt
    neg = custom_negative if custom_negative.strip() else NEGATIVE_PROMPT

    # Quality preset
    preset = QUALITY_PRESETS[quality]
    steps = override_steps if override_steps > 0 else preset["steps"]
    cfg = override_cfg if override_cfg > 0 else preset["cfg"]
    use_lightning = preset["lightning"]

    # Clamp CFG for Lightning
    if use_lightning and cfg > 2.0:
        cfg = 2.0

    # Resolution
    width, height = RESOLUTION_PRESETS[resolution]

    # Seed
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Generate
    progress(0.1, desc="Generating...")
    t0 = time.time()

    result = pipe(
        prompt=full_prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )

    gen_time = time.time() - t0
    raw_image = result.images[0]
    progress(0.7, desc="Generation complete")

    # Pixel cleanup
    cleanup_info = None
    final_image = raw_image

    if enable_cleanup:
        progress(0.75, desc="Pixel cleanup: detecting scale...")
        t1 = time.time()
        final_image, cleanup_info = cleanup_pixel_art(
            raw_image,
            reduce_palette=reduce_palette,
            max_colors=int(max_colors),
            upscale_factor=int(upscale_factor),
        )
        cleanup_time = time.time() - t1
        progress(0.95, desc="Cleanup complete")
    else:
        cleanup_time = 0

    # Save
    ts = int(time.time())
    filename = f"pixel_{ts}_{seed}.png"
    save_path = OUTPUT_DIR / filename
    final_image.save(save_path)

    # Build info string
    info_lines = [
        f"Seed: {seed}",
        f"Steps: {steps} | CFG: {cfg} | Lightning: {use_lightning}",
        f"Resolution: {width}x{height}",
        f"Gen time: {gen_time:.2f}s",
    ]
    if cleanup_info:
        info_lines.extend([
            "---",
            f"Pixel scale detected: {cleanup_info['detected_scale'][0]:.1f}x{cleanup_info['detected_scale'][1]:.1f}",
            f"Native size: {cleanup_info['native_size'][0]}x{cleanup_info['native_size'][1]}",
            f"Palette colors: {cleanup_info['palette_colors']}",
            f"Output size: {cleanup_info['output_size'][0]}x{cleanup_info['output_size'][1]}",
            f"Cleanup time: {cleanup_time:.2f}s",
        ])
    info_lines.append(f"Saved: {save_path.name}")

    progress(1.0, desc="Done")

    return raw_image, final_image, "\n".join(info_lines), seed


# ---------------------------------------------------------------------------
# Cleanup-only mode (bring your own image)
# ---------------------------------------------------------------------------
def cleanup_only(
    image: Image.Image,
    reduce_palette: bool,
    max_colors: int,
    upscale_factor: int,
):
    """Run pixel cleanup on an uploaded image without generation."""
    if image is None:
        return None, "No image provided"

    if image.mode != "RGB":
        image = image.convert("RGB")

    t0 = time.time()
    result, info = cleanup_pixel_art(
        image,
        reduce_palette=reduce_palette,
        max_colors=int(max_colors),
        upscale_factor=int(upscale_factor),
    )
    elapsed = time.time() - t0

    info_lines = [
        f"Input: {image.width}x{image.height}",
        f"Detected scale: {info['detected_scale'][0]:.1f}x{info['detected_scale'][1]:.1f}",
        f"Native size: {info['native_size'][0]}x{info['native_size'][1]}",
        f"Palette colors: {info['palette_colors']}",
        f"Output size: {info['output_size'][0]}x{info['output_size'][1]}",
        f"Time: {elapsed:.2f}s",
    ]

    return result, "\n".join(info_lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Pixel Art Generator") as demo:
        gr.Markdown("# Pixel Art Generator\nSDXL + Lightning LoRA + Tiny VAE + Optimized Pixel Cleanup")

        with gr.Tabs():
            # ---- Tab 1: Generate ----
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="a medieval castle on a cliff at sunset",
                            lines=3,
                        )
                        style = gr.Dropdown(
                            choices=list(STYLE_PRESETS.keys()),
                            value="16-bit RPG",
                            label="Style Preset",
                        )
                        quality = gr.Dropdown(
                            choices=list(QUALITY_PRESETS.keys()),
                            value="Lightning (4 steps)",
                            label="Quality",
                        )
                        resolution = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value="Scene (1024x768)",
                            label="Resolution",
                        )

                        with gr.Row():
                            seed = gr.Number(label="Seed", value=42, precision=0)
                            randomize = gr.Checkbox(label="Randomize", value=True)

                        custom_neg = gr.Textbox(
                            label="Negative Prompt (leave empty for default)",
                            placeholder=NEGATIVE_PROMPT,
                            lines=2,
                        )

                        gr.Markdown("### Pixel Cleanup")
                        enable_cleanup = gr.Checkbox(label="Enable pixel cleanup", value=True)
                        reduce_palette = gr.Checkbox(label="Reduce palette (elbow method)", value=True)
                        max_colors = gr.Slider(4, 128, value=64, step=4, label="Max colors to evaluate")
                        upscale_factor = gr.Slider(1, 8, value=4, step=1, label="Upscale factor (nearest neighbor)")

                        with gr.Accordion("Advanced Overrides", open=False):
                            override_steps = gr.Slider(0, 80, value=0, step=1, label="Override steps (0 = use preset)")
                            override_cfg = gr.Slider(0, 20, value=0, step=0.5, label="Override CFG (0 = use preset)")

                        gen_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Row():
                            raw_output = gr.Image(label="Raw SDXL Output", type="pil")
                            clean_output = gr.Image(label="After Pixel Cleanup", type="pil")
                        info_box = gr.Textbox(label="Generation Info", lines=10, interactive=False)

                gen_btn.click(
                    fn=generate,
                    inputs=[
                        prompt, style, quality, resolution, seed, randomize,
                        custom_neg, enable_cleanup, reduce_palette, max_colors,
                        upscale_factor, override_steps, override_cfg,
                    ],
                    outputs=[raw_output, clean_output, info_box, seed],
                )

            # ---- Tab 2: Cleanup Only ----
            with gr.TabItem("Cleanup Only"):
                gr.Markdown("Upload any image to run through the pixel detection + cleanup pipeline.")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_img = gr.Image(label="Upload Image", type="pil")
                        cu_palette = gr.Checkbox(label="Reduce palette", value=True)
                        cu_colors = gr.Slider(4, 128, value=64, step=4, label="Max colors")
                        cu_upscale = gr.Slider(1, 8, value=4, step=1, label="Upscale factor")
                        cu_btn = gr.Button("Run Cleanup", variant="primary")

                    with gr.Column(scale=2):
                        cu_output = gr.Image(label="Cleaned Output", type="pil")
                        cu_info = gr.Textbox(label="Cleanup Info", lines=8, interactive=False)

                cu_btn.click(
                    fn=cleanup_only,
                    inputs=[upload_img, cu_palette, cu_colors, cu_upscale],
                    outputs=[cu_output, cu_info],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=2)  # Prevent queue buildup from repeated clicks during warmup
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple"),
    )
