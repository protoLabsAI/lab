#!/usr/bin/env python3
"""
Pixel Art Generator — SDXL Lightning vs FLUX.2 Klein A/B test

Two model backends with shared pixel cleanup pipeline:
  - SDXL Lightning (4-step, fused LoRA, Tiny VAE) — current production
  - FLUX.2 Klein 4B (4-step distilled, Qwen3 text encoder) — experimental

Both support txt2img + img2img. Toggle between them to compare quality/speed.

Run: CUDA_VISIBLE_DEVICES=1 uv run python -u app.py
"""

import gc
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
    Flux2KleinPipeline,
    StableDiffusionXLImg2ImgPipeline,
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
OUTPUT_DIR = Path("/mnt/data/comfyui/output/pixel-gen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SDXL Lightning config
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TINY_VAE_ID = "madebyollin/taesdxl"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_FILE = "sdxl_lightning_4step_lora.safetensors"
SYSTEM_LORA_DIR = Path("/mnt/data/models-cold/system_loras")

# FLUX.2 Klein config
FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

MODEL_BACKENDS = ["SDXL Lightning", "FLUX.2 Klein 4B"]

RESOLUTION_PRESETS = {
    "Scene (1024x768)": (1024, 768),
    "Portrait (768x768)": (768, 768),
    "Icon (512x512)": (512, 512),
    "Wide (1024x576)": (1024, 576),
    "Square (1024x1024)": (1024, 1024),
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
# SDXL Lightning pipeline
# ---------------------------------------------------------------------------
_sdxl_pipe: StableDiffusionXLPipeline | None = None
_sdxl_img2img: StableDiffusionXLImg2ImgPipeline | None = None


def ensure_lightning_lora() -> Path | None:
    local = SYSTEM_LORA_DIR / LIGHTNING_FILE
    if local.exists():
        return local
    SYSTEM_LORA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return Path(hf_hub_download(
            repo_id=LIGHTNING_REPO, filename=LIGHTNING_FILE,
            local_dir=str(SYSTEM_LORA_DIR), local_dir_use_symlinks=False,
        ))
    except Exception as e:
        logger.error(f"Failed to download Lightning LoRA: {e}")
        return None


def get_sdxl_pipeline() -> StableDiffusionXLPipeline:
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    logger.info(f"Loading SDXL Lightning: {SDXL_MODEL_ID}")
    t0 = time.time()

    _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to(DEVICE)

    try:
        tiny_vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=torch.float16)
        _sdxl_pipe.vae = tiny_vae.to(DEVICE)
        logger.info("Tiny VAE loaded")
    except Exception as e:
        logger.warning(f"Tiny VAE failed: {e}")

    _sdxl_pipe.enable_vae_slicing()
    _sdxl_pipe.unet = _sdxl_pipe.unet.to(memory_format=torch.channels_last)
    _sdxl_pipe.vae = _sdxl_pipe.vae.to(memory_format=torch.channels_last)
    _sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(
        _sdxl_pipe.scheduler.config, timestep_spacing="trailing"
    )

    lightning_path = ensure_lightning_lora()
    if lightning_path:
        _sdxl_pipe.load_lora_weights(
            str(lightning_path.parent), weight_name=lightning_path.name,
            adapter_name="lightning",
        )
        _sdxl_pipe.set_adapters(["lightning"], [1.0])
        _sdxl_pipe.fuse_lora()
        _sdxl_pipe.unload_lora_weights()
        logger.info("Lightning LoRA fused")

    logger.info(f"SDXL Lightning loaded in {time.time() - t0:.1f}s")
    return _sdxl_pipe


def get_sdxl_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global _sdxl_img2img
    if _sdxl_img2img is not None:
        return _sdxl_img2img
    pipe = get_sdxl_pipeline()
    _sdxl_img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2, tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2, unet=pipe.unet, scheduler=pipe.scheduler,
    )
    logger.info("SDXL img2img created (shared weights)")
    return _sdxl_img2img


# ---------------------------------------------------------------------------
# FLUX.2 Klein pipeline
# ---------------------------------------------------------------------------
_flux_pipe: Flux2KleinPipeline | None = None


def get_flux_pipeline() -> Flux2KleinPipeline:
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe

    logger.info(f"Loading FLUX.2 Klein: {FLUX_MODEL_ID}")
    t0 = time.time()

    _flux_pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    logger.info(f"FLUX.2 Klein loaded in {time.time() - t0:.1f}s")
    return _flux_pipe


def unload_flux():
    """Free FLUX VRAM when switching back to SDXL."""
    global _flux_pipe
    if _flux_pipe is not None:
        del _flux_pipe
        _flux_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("FLUX.2 Klein unloaded")


def unload_sdxl():
    """Free SDXL VRAM when switching to FLUX."""
    global _sdxl_pipe, _sdxl_img2img
    if _sdxl_pipe is not None:
        del _sdxl_pipe, _sdxl_img2img
        _sdxl_pipe = None
        _sdxl_img2img = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("SDXL Lightning unloaded")


# ---------------------------------------------------------------------------
# Unified generation
# ---------------------------------------------------------------------------
def generate(
    prompt: str,
    model_backend: str,
    style: str,
    resolution: str,
    seed: int,
    randomize_seed: bool,
    custom_negative: str,
    use_img2img: bool,
    input_image: Image.Image | None,
    img2img_strength: float,
    enable_cleanup: bool,
    reduce_palette: bool,
    max_colors: int,
    upscale_factor: int,
    override_steps: int,
    override_cfg: float,
    progress=gr.Progress(track_tqdm=False),
):
    style_prefix = STYLE_PRESETS.get(style, "")
    full_prompt = f"{style_prefix}, {prompt}" if style_prefix else prompt
    neg = custom_negative if custom_negative.strip() else NEGATIVE_PROMPT

    width, height = RESOLUTION_PRESETS[resolution]

    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)

    is_flux = model_backend == "FLUX.2 Klein 4B"

    # Defaults per backend
    if is_flux:
        steps = override_steps if override_steps > 0 else 4
        cfg = override_cfg if override_cfg > 0 else 1.0
    else:
        steps = override_steps if override_steps > 0 else 4
        cfg = override_cfg if override_cfg > 0 else 1.5
        if cfg > 2.0:
            cfg = 2.0

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    progress(0.05, desc=f"Loading {model_backend}...")
    t0 = time.time()

    if is_flux:
        # Unload SDXL to free VRAM if loaded
        if _sdxl_pipe is not None:
            unload_sdxl()

        pipe = get_flux_pipeline()

        if use_img2img and input_image is not None:
            # FLUX Klein doesn't have an img2img pipeline — fall back to txt2img
            logger.warning("FLUX.2 Klein does not support img2img, running txt2img instead")
            mode_label = "FLUX txt2img (img2img not supported)"
        else:
            mode_label = "FLUX txt2img"

        progress(0.1, desc="FLUX generating...")
        result = pipe(
            prompt=full_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
    else:
        # Unload FLUX to free VRAM if loaded
        if _flux_pipe is not None:
            unload_flux()

        if use_img2img and input_image is not None:
            pipe = get_sdxl_img2img()
            init = input_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            progress(0.1, desc="SDXL img2img generating...")
            result = pipe(
                prompt=full_prompt,
                negative_prompt=neg,
                image=init,
                strength=img2img_strength,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
            mode_label = f"SDXL img2img (strength={img2img_strength})"
        else:
            pipe = get_sdxl_pipeline()
            progress(0.1, desc="SDXL generating...")
            result = pipe(
                prompt=full_prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
            mode_label = "SDXL txt2img"

    gen_time = time.time() - t0
    raw_image = result.images[0]
    progress(0.7, desc="Generation complete")

    # Pixel cleanup (shared across both backends)
    cleanup_info = None
    final_image = raw_image
    cleanup_time = 0

    if enable_cleanup:
        progress(0.75, desc="Pixel cleanup...")
        t1 = time.time()
        final_image, cleanup_info = cleanup_pixel_art(
            raw_image,
            reduce_palette=reduce_palette,
            max_colors=int(max_colors),
            upscale_factor=int(upscale_factor),
        )
        cleanup_time = time.time() - t1
        progress(0.95, desc="Cleanup complete")

    # Save
    ts = int(time.time())
    filename = f"pixel_{ts}_{seed}.png"
    save_path = OUTPUT_DIR / filename
    final_image.save(save_path)

    # Info
    info_lines = [
        f"Backend: {model_backend}",
        f"Mode: {mode_label}",
        f"Seed: {seed}",
        f"Steps: {steps} | CFG: {cfg}",
        f"Resolution: {width}x{height}",
        f"Gen time: {gen_time:.2f}s",
    ]
    if cleanup_info:
        info_lines.extend([
            "---",
            f"Native: {cleanup_info['native_size'][0]}x{cleanup_info['native_size'][1]}",
            f"Palette: {cleanup_info['palette_colors']} colors",
            f"Cleanup: {cleanup_time:.2f}s",
        ])
    info_lines.append(f"Saved: {save_path.name}")

    progress(1.0, desc="Done")
    return raw_image, final_image, "\n".join(info_lines), seed


# ---------------------------------------------------------------------------
# Cleanup-only
# ---------------------------------------------------------------------------
def cleanup_only(image, reduce_palette, max_colors, upscale_factor):
    if image is None:
        return None, "No image provided"
    if image.mode != "RGB":
        image = image.convert("RGB")
    t0 = time.time()
    result, info = cleanup_pixel_art(image, reduce_palette, int(max_colors), int(upscale_factor))
    elapsed = time.time() - t0
    return result, "\n".join([
        f"Input: {image.width}x{image.height}",
        f"Native: {info['native_size'][0]}x{info['native_size'][1]}",
        f"Palette: {info['palette_colors']} colors",
        f"Output: {info['output_size'][0]}x{info['output_size'][1]}",
        f"Time: {elapsed:.2f}s",
    ])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Pixel Art Generator") as demo:
        gr.Markdown(
            "# Pixel Art Generator\n"
            "SDXL Lightning vs FLUX.2 Klein 4B — same pixel cleanup, compare quality + speed"
        )

        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_backend = gr.Radio(
                            choices=MODEL_BACKENDS,
                            value="SDXL Lightning",
                            label="Model Backend",
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="a medieval castle on a cliff at sunset",
                            lines=3,
                        )
                        style = gr.Dropdown(
                            choices=list(STYLE_PRESETS.keys()),
                            value="16-bit RPG", label="Style Preset",
                        )
                        resolution = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value="Scene (1024x768)", label="Resolution",
                        )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=42, precision=0)
                            randomize = gr.Checkbox(label="Randomize", value=True)
                        custom_neg = gr.Textbox(
                            label="Negative Prompt (leave empty for default)",
                            placeholder=NEGATIVE_PROMPT, lines=2,
                        )

                        gr.Markdown("### Image Input (img2img)")
                        use_img2img = gr.Checkbox(label="Use reference image", value=False)
                        input_image = gr.Image(
                            label="Reference Image", type="pil", visible=False,
                        )
                        img2img_strength = gr.Slider(
                            0.25, 1.0, value=0.5, step=0.25,
                            label="Strength",
                            info="0.25 = subtle, 0.50 = balanced, 0.75 = creative",
                            visible=False,
                        )

                        use_img2img.change(
                            fn=lambda v: (gr.update(visible=v), gr.update(visible=v)),
                            inputs=[use_img2img],
                            outputs=[input_image, img2img_strength],
                        )

                        gr.Markdown("### Pixel Cleanup")
                        enable_cleanup = gr.Checkbox(label="Enable pixel cleanup", value=True)
                        reduce_palette = gr.Checkbox(label="Reduce palette", value=True)
                        max_colors = gr.Slider(4, 128, value=64, step=4, label="Max colors")
                        upscale_factor = gr.Slider(1, 8, value=4, step=1, label="Upscale factor")

                        with gr.Accordion("Advanced Overrides", open=False):
                            override_steps = gr.Slider(0, 50, value=0, step=1, label="Override steps (0 = default)")
                            override_cfg = gr.Slider(0, 20, value=0, step=0.5, label="Override CFG (0 = default)")

                        gen_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Row():
                            raw_output = gr.Image(label="Raw Output", type="pil")
                            clean_output = gr.Image(label="After Pixel Cleanup", type="pil")
                        info_box = gr.Textbox(label="Generation Info", lines=12, interactive=False)

                gen_btn.click(
                    fn=generate,
                    inputs=[
                        prompt, model_backend, style, resolution, seed, randomize,
                        custom_neg, use_img2img, input_image, img2img_strength,
                        enable_cleanup, reduce_palette, max_colors, upscale_factor,
                        override_steps, override_cfg,
                    ],
                    outputs=[raw_output, clean_output, info_box, seed],
                )

            with gr.TabItem("Cleanup Only"):
                gr.Markdown("Upload any image to run through the pixel cleanup pipeline.")
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
    demo.queue(max_size=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple"),
    )
