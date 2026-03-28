#!/usr/bin/env python3
"""
Speech-to-Text — Whisper model comparison + speaker diarization

Compare whisper-large-v3-turbo vs large-v3 vs distil-large-v3 on Blackwell.
Upload audio or record from mic, get transcription with speed metrics.
Optional speaker diarization via pyannote-audio.

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
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/mnt/data/comfyui/output/stt-whisper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.float16

HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODELS = {
    "large-v3-turbo (809M, fastest)": "openai/whisper-large-v3-turbo",
    "distil-large-v3 (756M, fast)": "distil-whisper/distil-large-v3",
    "large-v3 (1.55B, best quality)": "openai/whisper-large-v3",
}

_pipes: dict[str, object] = {}
_current_model: str | None = None
_diarization_pipeline = None


def get_pipe(model_key: str):
    global _current_model
    model_id = MODELS[model_key]

    if model_key in _pipes:
        return _pipes[model_key]

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


def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    if not HF_TOKEN:
        return None

    logger.info("Loading pyannote diarization pipeline...")
    t0 = time.time()
    from pyannote.audio import Pipeline as PyannotePipeline

    _diarization_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    ).to(torch.device(DEVICE))

    logger.info(f"Diarization pipeline loaded in {time.time() - t0:.1f}s")
    return _diarization_pipeline


def merge_diarization(transcript_chunks, diarization_result):
    """Merge whisper transcript segments with pyannote speaker labels."""
    lines = []
    for chunk in transcript_chunks:
        ts = chunk.get("timestamp", (None, None))
        start = ts[0] if ts[0] is not None else 0.0

        # Find which speaker is talking at this timestamp
        speaker = "?"
        for turn, _, spk in diarization_result.itertracks(yield_label=True):
            if turn.start <= start < turn.end:
                speaker = spk
                break

        start_str = f"{start:.1f}s" if ts[0] is not None else ""
        lines.append(f"[{speaker}] {start_str}: {chunk['text'].strip()}")

    return "\n".join(lines)


def transcribe(
    audio,
    model_key: str,
    batch_size: int,
    return_timestamps: bool,
    enable_diarization: bool,
    language: str,
):
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

    transcribe_time = time.time() - t0
    text = result["text"]

    # Audio duration
    audio_data, sr = sf.read(audio)
    duration = len(audio_data) / sr

    # Diarization
    diarize_time = 0
    diarized_text = None
    if enable_diarization and "chunks" in result:
        diar_pipe = get_diarization_pipeline()
        if diar_pipe is None:
            diarized_text = "(Diarization requires HF_TOKEN env var — pyannote models are gated)"
        else:
            t1 = time.time()
            diar_result = diar_pipe(audio)
            diarize_time = time.time() - t1
            diarized_text = merge_diarization(result["chunks"], diar_result)

    total_time = transcribe_time + diarize_time
    rtf = total_time / duration if duration > 0 else 0
    speed_x = duration / total_time if total_time > 0 else 0

    # Save
    ts = int(time.time())
    save_path = OUTPUT_DIR / f"transcript_{ts}.txt"
    save_text = diarized_text if diarized_text else text
    save_path.write_text(save_text)

    info_lines = [
        f"Model: {model_id}",
        f"Audio: {duration:.1f}s",
        f"Transcription: {transcribe_time:.2f}s",
    ]
    if diarize_time > 0:
        info_lines.append(f"Diarization: {diarize_time:.2f}s")
    info_lines.extend([
        f"Total: {total_time:.2f}s",
        f"Speed: {speed_x:.1f}x real-time",
        f"RTF: {rtf:.4f}",
        f"Batch size: {batch_size}",
    ])
    if return_timestamps and "chunks" in result:
        info_lines.append(f"Segments: {len(result['chunks'])}")
    info_lines.append(f"Saved: {save_path.name}")

    display_text = diarized_text if diarized_text else text
    return display_text, "\n".join(info_lines)


def build_ui():
    with gr.Blocks(title="Speech-to-Text") as demo:
        gr.Markdown(
            "# Speech-to-Text — Whisper + Diarization\n"
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
                    info="Higher = faster but more VRAM",
                )
                timestamps = gr.Checkbox(
                    label="Word-level timestamps",
                    value=False,
                )
                diarization = gr.Checkbox(
                    label="Speaker diarization (pyannote)",
                    value=False,
                    info="Requires HF_TOKEN env var" if not HF_TOKEN else "Ready",
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=2):
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=15,
                )
                info_output = gr.Textbox(
                    label="Performance Info",
                    lines=8,
                    interactive=False,
                )

        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, model_select, batch_size, timestamps, diarization, language],
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
