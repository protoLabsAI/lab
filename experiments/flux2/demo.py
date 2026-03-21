#!/usr/bin/env python3
"""
FLUX.2 Klein 9B — Distilled Gradio Demo
Run: CUDA_VISIBLE_DEVICES=1 python -u demo.py
"""

import os
import gc
import random
import time
from pathlib import Path

# Route HuggingFace to the models drive
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import numpy as np
import torch
import gradio as gr
from PIL import Image
from diffusers import Flux2KleinPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = Path("/mnt/data/comfyui/output/flux2")

# Resolution presets: (width, height)
RESOLUTIONS = {
    "1024 × 1024 (1:1)": (1024, 1024),
    "1280 × 768 (5:3)": (1280, 768),
    "768 × 1280 (3:5)": (768, 1280),
    "1024 × 576 (16:9)": (1024, 576),
    "576 × 1024 (9:16)": (576, 1024),
}
DEFAULT_RESOLUTION = "1024 × 1024 (1:1)"

MODES = {
    "Distilled (4 steps)": {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "steps": 4,
        "guidance": 1.0,
    },
    "Base (50 steps)": {
        "repo": "black-forest-labs/FLUX.2-klein-base-9B",
        "steps": 50,
        "guidance": 4.0,
    },
}

# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
pipelines: dict[str, Flux2KleinPipeline] = {}


def get_pipeline(mode: str) -> Flux2KleinPipeline:
    """Load pipeline on first use, keep in VRAM."""
    if mode not in pipelines:
        cfg = MODES[mode]
        print(f"Loading {mode} from {cfg['repo']}...")
        pipe = Flux2KleinPipeline.from_pretrained(cfg["repo"], torch_dtype=DTYPE)
        pipe.to(DEVICE)
        pipelines[mode] = pipe
        print(f"{mode} ready.")
    return pipelines[mode]


def log_vram(tag: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(
            f"[VRAM {tag}] alloc={alloc:.1f}GB  peak={peak:.1f}GB  "
            f"free={free / 1024**3:.1f}GB  total={total / 1024**3:.1f}GB"
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate(
    prompt: str,
    input_images: list | None,
    mode_choice: str,
    resolution: str,
    seed: int,
    randomize_seed: bool,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt.strip():
        gr.Warning("Please enter a prompt.")
        return None, 0

    pipe = get_pipeline(mode_choice)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    width, height = RESOLUTIONS[resolution]

    torch.cuda.reset_peak_memory_stats()
    log_vram("start")

    print(f"Generating: {width}×{height}, {num_inference_steps} steps, "
          f"cfg={guidance_scale}, seed={seed}")

    t0 = time.perf_counter()

    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": torch.Generator(device=DEVICE).manual_seed(seed),
    }

    # Multi-image editing support
    if input_images and len(input_images) > 0:
        images = []
        for item in input_images:
            img = item[0] if isinstance(item, tuple) else item
            if isinstance(img, str):
                img = Image.open(img)
            images.append(img)
        pipe_kwargs["image"] = images

    image = pipe(**pipe_kwargs).images[0]

    elapsed = time.perf_counter() - t0
    log_vram("done")
    print(f"Generated in {elapsed:.1f}s")

    # Save to output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"flux2_{seed}.png"
    image.save(out_path)

    return image, seed


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------
def on_mode_change(mode_choice: str):
    cfg = MODES[mode_choice]
    return cfg["steps"], cfg["guidance"]


def on_image_upload(image_list):
    """Auto-detect aspect ratio from first uploaded image."""
    if not image_list or len(image_list) == 0:
        return gr.update()

    img = image_list[0][0] if isinstance(image_list[0], tuple) else image_list[0]
    if isinstance(img, str):
        img = Image.open(img)
    w, h = img.size
    ratio = w / h

    # Find closest resolution preset
    best = DEFAULT_RESOLUTION
    best_diff = float("inf")
    for name, (rw, rh) in RESOLUTIONS.items():
        diff = abs(ratio - rw / rh)
        if diff < best_diff:
            best_diff = diff
            best = name

    return gr.update(value=best)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
css = """
#col-container { margin: 0 auto; max-width: 1200px; }
"""

with gr.Blocks(title="FLUX.2 Klein 9B") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            "# FLUX.2 \\[Klein\\] 9B — Local\n"
            "Full bf16 on Blackwell — no quantization, no CPU offload."
        )

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Describe the image you want to generate...",
                    autofocus=True,
                )

                with gr.Accordion("Input Images (optional — for editing/combining)", open=False):
                    input_images = gr.Gallery(
                        label="Reference Images",
                        type="pil",
                        columns=3,
                        rows=1,
                    )

                with gr.Row():
                    mode_choice = gr.Radio(
                        label="Mode",
                        choices=list(MODES.keys()),
                        value="Distilled (4 steps)",
                    )
                    resolution = gr.Dropdown(
                        label="Resolution",
                        choices=list(RESOLUTIONS.keys()),
                        value=DEFAULT_RESOLUTION,
                    )

                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Inference steps",
                            minimum=1, maximum=100, step=1, value=4,
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0, maximum=10.0, step=0.1, value=1.0,
                        )

            with gr.Column():
                result = gr.Image(label="Result", show_label=False)

        # Events
        mode_choice.change(
            fn=on_mode_change,
            inputs=[mode_choice],
            outputs=[num_inference_steps, guidance_scale],
        )

        input_images.upload(
            fn=on_image_upload,
            inputs=[input_images],
            outputs=[resolution],
        )

        gr.on(
            triggers=[generate_btn.click, prompt.submit],
            fn=generate,
            inputs=[
                prompt, input_images, mode_choice, resolution,
                seed, randomize_seed, num_inference_steps, guidance_scale,
            ],
            outputs=[result, seed],
        )


if __name__ == "__main__":
    # Eager-load distilled model at startup
    print("Pre-loading distilled pipeline...")
    get_pipeline("Distilled (4 steps)")
    print("Ready.")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        css=css,
        allowed_paths=[str(OUTPUT_DIR)],
    )
