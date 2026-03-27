#!/usr/bin/env python3
"""
Speech-to-Text — Whisper model comparison

Compare whisper-large-v3-turbo vs large-v3 vs distil-large-v3 on Blackwell.
Upload audio or record from mic, get transcription with speed metrics.

Uses HuggingFace Transformers pipeline (SDPA attention, no flash-attn needed).

Run: CUDA_VISIBLE_DEVICES=1 uv run python -u app.py
"""

import gc
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import gradio as gr
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/mnt/data/comfyui/output/stt-whisper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.float16

MODELS = {
    "large-v3-turbo (809M, fastest)": "openai/whisper-large-v3-turbo",
    "distil-large-v3 (756M, fast)": "distil-whisper/distil-large-v3",
    "large-v3 (1.55B, best quality)": "openai/whisper-large-v3",
}

# Cache loaded pipelines
_pipes: dict[str, object] = {}
_current_model: str | None = None


def get_pipe(model_key: str):
    """Load or return cached pipeline. Unloads previous model to save VRAM."""
    global _current_model

    model_id = MODELS[model_key]

    if model_key in _pipes:
        return _pipes[model_key]

    # Unload previous model
    if _current_model and _current_model != model_key and _current_model in _pipes:
        del _pipes[_current_model]
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Unloaded {_current_model}")

    logger.info(f"Loading {model_id}...")
    t0 = time.time()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=DTYPE,
        device=DEVICE,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    _pipes[model_key] = pipe
    _current_model = model_key
    logger.info(f"Loaded {model_id} in {time.time() - t0:.1f}s")
    return pipe


def transcribe(
    audio,
    model_key: str,
    batch_size: int,
    return_timestamps: bool,
    language: str,
):
    """Transcribe audio file or recording."""
    if audio is None:
        return "", "No audio provided"

    pipe = get_pipe(model_key)
    model_id = MODELS[model_key]

    generate_kwargs = {}
    if language and language != "auto":
        generate_kwargs["language"] = language

    t0 = time.time()

    result = pipe(
        audio,
        chunk_length_s=30,
        batch_size=batch_size,
        return_timestamps="word" if return_timestamps else True,
        generate_kwargs=generate_kwargs,
    )

    elapsed = time.time() - t0

    text = result["text"]

    # Calculate audio duration
    import soundfile as sf
    audio_data, sr = sf.read(audio)
    duration = len(audio_data) / sr

    rtf = elapsed / duration if duration > 0 else 0
    speed_x = duration / elapsed if elapsed > 0 else 0

    # Save transcript
    ts = int(time.time())
    save_path = OUTPUT_DIR / f"transcript_{ts}.txt"
    save_path.write_text(text)

    info_lines = [
        f"Model: {model_id}",
        f"Audio: {duration:.1f}s",
        f"Transcription time: {elapsed:.2f}s",
        f"Speed: {speed_x:.1f}x real-time",
        f"RTF: {rtf:.4f}",
        f"Batch size: {batch_size}",
        f"Saved: {save_path.name}",
    ]

    # Add timestamps if requested
    if return_timestamps and "chunks" in result:
        info_lines.append(f"Segments: {len(result['chunks'])}")

    return text, "\n".join(info_lines)


def build_ui():
    with gr.Blocks(title="Speech-to-Text") as demo:
        gr.Markdown(
            "# Speech-to-Text — Whisper Comparison\n"
            "Compare large-v3-turbo vs distil-large-v3 vs large-v3 on Blackwell"
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload or Record Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                model_select = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="large-v3-turbo (809M, fastest)",
                    label="Model",
                )
                language = gr.Dropdown(
                    choices=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "ja", "zh", "ko", "ar", "hi"],
                    value="auto",
                    label="Language",
                )
                batch_size = gr.Slider(
                    1, 48, value=24, step=4,
                    label="Batch Size",
                    info="Higher = faster but more VRAM. 24 is good for 96GB.",
                )
                timestamps = gr.Checkbox(
                    label="Word-level timestamps",
                    value=False,
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=2):
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=12,
                )
                info_output = gr.Textbox(
                    label="Performance Info",
                    lines=8,
                    interactive=False,
                )

        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, model_select, batch_size, timestamps, language],
            outputs=[transcript_output, info_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
