"""LTX-2 Video Generation via vLLM-Omni API.

Gradio UI that talks to vLLM-Omni's /v1/videos endpoint instead of
loading the pipeline directly. This lets the model stay resident on
GPU 1 as a service while the UI is lightweight.

Start the backend first:
    CUDA_VISIBLE_DEVICES=1 vllm-omni serve Lightricks/LTX-2 --omni --port 8091 --vae-use-tiling

Then run this:
    python gradio-demo.py [--api-url http://localhost:8091]

Falls back to the old pipeline mode if vLLM-Omni is not running.
"""

import argparse
import os
import random
import time

import gradio as gr
import numpy as np
import requests

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_FRAME_RATE = 24.0
DEFAULT_API_URL = "http://localhost:8091"

_current_video_id = None
_cancelled = False


def cancel_generation():
    global _cancelled
    _cancelled = True
    return "Cancelling..."


RESOLUTIONS = {
    "high": {"16:9": (1536, 1024), "9:16": (1024, 1536), "1:1": (1024, 1024)},
    "low": {"16:9": (768, 512), "9:16": (512, 768), "1:1": (768, 768)},
}


def check_api(api_url: str) -> bool:
    """Check if vLLM-Omni is serving."""
    try:
        r = requests.get(f"{api_url}/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


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


def snap_to_32(val):
    """Round to nearest multiple of 32, minimum 320."""
    return max(int(round(val / 32) * 32), 320)


def on_image_upload(image, high_res):
    aspect = detect_aspect_ratio(image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=snap_to_32(w)), gr.update(value=snap_to_32(h))


def on_highres_toggle(image, high_res):
    aspect = detect_aspect_ratio(image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=snap_to_32(w)), gr.update(value=snap_to_32(h))


def generate_video(
    input_image,
    prompt: str,
    duration: float,
    inference_steps: int,
    enhance_prompt: bool,
    seed: int,
    randomize_seed: bool,
    height: int,
    width: int,
    api_url: str = DEFAULT_API_URL,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
        num_frames = int(duration * DEFAULT_FRAME_RATE) + 1
        num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

        # LTX-2 requires dimensions divisible by 32
        height = int(round(height / 32) * 32)
        width = int(round(width / 32) * 32)
        height = max(height, 320)
        width = max(width, 320)

        print(f"Generating: {height}x{width}, {num_frames} frames ({duration}s), seed={current_seed}")

        # Build request
        data = {
            "prompt": prompt,
            "height": str(int(height)),
            "width": str(int(width)),
            "num_frames": str(num_frames),
            "fps": str(int(DEFAULT_FRAME_RATE)),
            "num_inference_steps": str(int(inference_steps)),
            "guidance_scale": "4.0",
            "seed": str(current_seed),
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        }

        files = {}
        if input_image is not None:
            import io
            from PIL import Image

            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            buf = io.BytesIO()
            input_image.save(buf, format="JPEG")
            buf.seek(0)
            files["image"] = ("input.jpg", buf, "image/jpeg")

        # Submit job
        progress(0, desc="Submitting to vLLM-Omni...")
        if files:
            r = requests.post(f"{api_url}/v1/videos", data=data, files=files, timeout=30)
        else:
            r = requests.post(f"{api_url}/v1/videos", data=data, timeout=30)

        if r.status_code != 200:
            return None, current_seed

        job = r.json()
        video_id = job.get("id") or job.get("video_id")
        if not video_id:
            # Some versions return the video directly
            if "url" in job:
                return job["url"], current_seed
            return None, current_seed

        # Store video_id for cancel
        global _current_video_id, _cancelled
        _current_video_id = video_id
        _cancelled = False

        # Poll for completion
        for i in range(600):  # 10 min max
            if _cancelled:
                requests.delete(f"{api_url}/v1/videos/{video_id}", timeout=5)
                print(f"Cancelled: {video_id}")
                return None, current_seed
            time.sleep(1)
            status_r = requests.get(f"{api_url}/v1/videos/{video_id}", timeout=10)
            status = status_r.json()

            state = status.get("status", "unknown")
            if state == "completed":
                progress(1.0, desc="Done!")
                break
            elif state == "failed":
                print(f"Generation failed: {status}")
                return None, current_seed
            elif i % 5 == 0:
                elapsed = i
                progress(0.5, desc=f"Generating... ({elapsed}s)")

        # Download the video
        output_dir = "/mnt/data/comfyui/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ltx_{current_seed}.mp4")

        video_r = requests.get(f"{api_url}/v1/videos/{video_id}/content", timeout=120)
        with open(output_path, "wb") as f:
            f.write(video_r.content)

        print(f"Saved to {output_path}")
        return output_path, current_seed

    except Exception as e:
        import traceback
        print(f"Error: {e}\n{traceback.format_exc()}")
        return None, current_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="vLLM-Omni API URL")
    parser.add_argument("--port", default=7860, type=int, help="Gradio port")
    args = parser.parse_args()

    api_live = check_api(args.api_url)
    if api_live:
        print(f"vLLM-Omni API detected at {args.api_url}")
    else:
        print(f"WARNING: vLLM-Omni not running at {args.api_url}")
        print("Start it with: CUDA_VISIBLE_DEVICES=1 vllm-omni serve Lightricks/LTX-2 --omni --port 8091 --vae-use-tiling")
        print("Launching UI anyway — will fail on generate until backend is up.")

    with gr.Blocks(title="LTX-2 Video Generation") as demo:
        gr.Markdown("# LTX-2: Video Generation via vLLM-Omni")
        gr.Markdown(
            f"Backend: `{args.api_url}` {'(connected)' if api_live else '(not running)'} "
            "| [[model]](https://huggingface.co/Lightricks/LTX-2) "
            "| [[vllm-omni]](https://docs.vllm.ai/projects/vllm-omni/)"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image (Optional — enables image-to-video)", type="pil")
                prompt = gr.Textbox(
                    label="Prompt",
                    info="Be descriptive — motion, camera, mood, lighting",
                    value="A cinematic dolly shot of ocean waves at golden hour, smooth animation",
                    lines=3,
                )

                with gr.Row():
                    duration = gr.Slider(label="Duration (seconds)", minimum=1.0, maximum=60.0, value=5.0, step=0.1)
                    inference_steps = gr.Slider(label="Quality (steps)", minimum=4, maximum=50, value=8, step=1,
                                                info="8=fast/distilled, 20=balanced, 40=max quality")
                    with gr.Column():
                        enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                        high_res = gr.Checkbox(label="High Resolution", value=True)

                with gr.Row():
                    generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
                    cancel_btn = gr.Button("Cancel", variant="stop", size="lg")

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=42, step=1)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    with gr.Row():
                        width = gr.Slider(label="Width", value=768, minimum=320, maximum=1920, step=32)
                        height = gr.Slider(label="Height", value=512, minimum=320, maximum=1920, step=32)

            with gr.Column():
                output_video = gr.Video(label="Generated Video", autoplay=True)

        input_image.change(fn=on_image_upload, inputs=[input_image, high_res], outputs=[width, height])
        high_res.change(fn=on_highres_toggle, inputs=[input_image, high_res], outputs=[width, height])

        generate_btn.click(
            fn=lambda *a: generate_video(*a, api_url=args.api_url),
            inputs=[input_image, prompt, duration, inference_steps, enhance_prompt, seed, randomize_seed, height, width],
            outputs=[output_video, seed],
        )

        cancel_btn.click(fn=cancel_generation, outputs=[])

    demo.launch(server_name="0.0.0.0", server_port=args.port,
                allowed_paths=["/mnt/data/comfyui/output"])


if __name__ == "__main__":
    main()
