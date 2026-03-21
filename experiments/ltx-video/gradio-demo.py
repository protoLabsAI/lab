import os
import sys
import logging
import random
import tempfile
from pathlib import Path

# Disable torch.compile / dynamo
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Use our local LTX-2 repo
LTX_REPO_DIR = os.path.expanduser("~/dev/LTX-2")
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

# Force pytorch native attention — xformers doesn't support Blackwell (compute 12.0)
from ltx_core.model.transformer import attention as _attn_mod
_attn_mod.memory_efficient_attention = None

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

logging.getLogger().setLevel(logging.INFO)

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_FRAME_RATE = 24.0

# Local model paths
LTX_SNAP = "/mnt/models/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/cd784f3198e9ec3efec60b66a0fd78aafe413a86"
GEMMA_ROOT = "/mnt/models/gemma-3-12b"

checkpoint_path = os.path.join(LTX_SNAP, "ltx-2.3-22b-distilled.safetensors")
spatial_upsampler_path = os.path.join(LTX_SNAP, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

# Resolution presets: (width, height)
RESOLUTIONS = {
    "high": {"16:9": (1536, 1024), "9:16": (1024, 1536), "1:1": (1024, 1024)},
    "low": {"16:9": (768, 512), "9:16": (512, 768), "1:1": (768, 768)},
}

print("=" * 80)
print("Initializing LTX-2.3 Distilled Pipeline...")
print(f"Checkpoint: {checkpoint_path}")
print(f"Spatial upsampler: {spatial_upsampler_path}")
print(f"Gemma root: {GEMMA_ROOT}")
print("=" * 80)

pipeline = DistilledPipeline(
    distilled_checkpoint_path=checkpoint_path,
    spatial_upsampler_path=spatial_upsampler_path,
    gemma_root=GEMMA_ROOT,
    loras=[],
    quantization=QuantizationPolicy.fp8_cast(),
)

# Models load on-demand during first generation
print("Pipeline ready — models will load on first use.")
print("=" * 80)


def log_memory(tag: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(f"[VRAM {tag}] allocated={allocated:.2f}GB peak={peak:.2f}GB free={free / 1024**3:.2f}GB total={total / 1024**3:.2f}GB")


def detect_aspect_ratio(image) -> str:
    if image is None:
        return "16:9"
    if hasattr(image, "size"):
        w, h = image.size
    elif hasattr(image, "shape"):
        h, w = image.shape[:2]
    else:
        return "16:9"
    ratio = w / h
    candidates = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
    return min(candidates, key=lambda k: abs(ratio - candidates[k]))


def on_image_upload(image, high_res):
    aspect = detect_aspect_ratio(image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=w), gr.update(value=h)


def on_highres_toggle(image, high_res):
    aspect = detect_aspect_ratio(image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=w), gr.update(value=h)


@torch.inference_mode()
def generate_video(
    input_image,
    prompt: str,
    duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = 1024,
    width: int = 1536,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        torch.cuda.reset_peak_memory_stats()
        log_memory("start")

        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

        frame_rate = DEFAULT_FRAME_RATE
        num_frames = int(duration * frame_rate) + 1
        num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

        print(f"Generating: {height}x{width}, {num_frames} frames ({duration}s), seed={current_seed}")

        images = []
        if input_image is not None:
            output_dir = Path("/mnt/data/comfyui/output")
            output_dir.mkdir(exist_ok=True)
            temp_image_path = output_dir / f"temp_input_{current_seed}.jpg"
            if hasattr(input_image, "save"):
                input_image.save(temp_image_path)
            else:
                temp_image_path = Path(input_image)
            images = [ImageConditioningInput(path=str(temp_image_path), frame_idx=0, strength=1.0)]

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        log_memory("before pipeline call")

        video, audio = pipeline(
            prompt=prompt,
            seed=current_seed,
            height=int(height),
            width=int(width),
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        log_memory("after pipeline call")

        output_path = os.path.join("/mnt/data/comfyui/output", f"ltx_{current_seed}.mp4")
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        log_memory("after encode_video")
        return str(output_path), current_seed

    except Exception as e:
        import traceback
        log_memory("on error")
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return None, current_seed


with gr.Blocks(title="LTX-2.3 Distilled") as demo:
    gr.Markdown("# LTX-2.3 Distilled (22B): Fast Audio-Video Generation")
    gr.Markdown(
        "Fast and high quality video + audio generation "
        "[[model]](https://huggingface.co/Lightricks/LTX-2.3) "
        "[[code]](https://github.com/Lightricks/LTX-2)"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image (Optional)", type="pil")
            prompt = gr.Textbox(
                label="Prompt",
                info="for best results - make it as elaborate as possible",
                value="Make this image come alive with cinematic motion, smooth animation",
                lines=3,
                placeholder="Describe the motion and animation you want...",
            )

            with gr.Row():
                duration = gr.Slider(label="Duration (seconds)", minimum=1.0, maximum=60.0, value=5.0, step=0.1)
                with gr.Column():
                    enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                    high_res = gr.Checkbox(label="High Resolution", value=True)

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=10, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                with gr.Row():
                    width = gr.Number(label="Width", value=1536, precision=0)
                    height = gr.Number(label="Height", value=1024, precision=0)

        with gr.Column():
            output_video = gr.Video(label="Generated Video", autoplay=True)

    input_image.change(
        fn=on_image_upload,
        inputs=[input_image, high_res],
        outputs=[width, height],
    )

    high_res.change(
        fn=on_highres_toggle,
        inputs=[input_image, high_res],
        outputs=[width, height],
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            input_image, prompt, duration, enhance_prompt,
            seed, randomize_seed, height, width,
        ],
        outputs=[output_video, seed],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Citrus(), allowed_paths=["/mnt/data/comfyui/output"])
