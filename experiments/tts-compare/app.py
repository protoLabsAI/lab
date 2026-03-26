#!/usr/bin/env python3
"""
TTS A/B/C Comparison — Voxtral 4B vs Fish Audio S2 Pro vs Kokoro 82M

Voxtral + Fish Audio run as external services (GPU-heavy).
Kokoro runs in-process (82M params, ~2GB VRAM, trivial).

Run: uv run python -u app.py
"""

import io
import logging
import os
import time
from pathlib import Path

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/mnt/data/comfyui/output/tts-compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOXTRAL_URL = os.environ.get("VOXTRAL_URL", "http://localhost:8091")
FISH_URL = os.environ.get("FISH_URL", "http://localhost:8092")

VOXTRAL_VOICES = [
    "casual_male", "casual_female", "neutral_male", "neutral_female",
    "cheerful_female", "ar_male",
    "de_female", "de_male", "es_female", "es_male",
    "fr_female", "fr_male", "hi_female", "hi_male",
    "it_female", "it_male", "nl_female", "nl_male",
    "pt_female", "pt_male",
]

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

SAMPLE_TEXTS = {
    "Short": "The quick brown fox jumps over the lazy dog.",
    "Medium": "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat. It was a hobbit-hole, and that means comfort.",
    "Dialogue": "Welcome, traveler. The road ahead is dangerous, but I sense you are no ordinary wanderer. Take this sword — it was forged in the fires of Mount Ashvale. May it serve you well.",
    "Emotional": "I can't believe it's finally over. After all these years of searching, we found it. We actually found it! The tears streamed down her face as she held the ancient artifact in trembling hands.",
    "Technical": "The transformer architecture uses self-attention mechanisms to process input sequences in parallel, achieving state-of-the-art results on natural language processing benchmarks.",
}

# ---------------------------------------------------------------------------
# Kokoro — in-process, lazy loaded
# ---------------------------------------------------------------------------
_kokoro_pipeline = None


def get_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        logger.info("Loading Kokoro 82M...")
        t0 = time.time()
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")  # American English
        logger.info(f"Kokoro loaded in {time.time() - t0:.1f}s")
    return _kokoro_pipeline


def generate_kokoro(text: str, voice: str) -> tuple[str | None, str]:
    """Generate audio via Kokoro 82M (in-process). Returns file path."""
    t0 = time.time()
    try:
        pipe = get_kokoro()
        chunks = list(pipe(text, voice=voice, speed=1))
        if not chunks:
            return None, "Kokoro: no audio generated"

        audio = np.concatenate([c[2] for c in chunks])
        sr = 24000
        elapsed = time.time() - t0
        duration = len(audio) / sr

        ts = int(time.time())
        save_path = OUTPUT_DIR / f"kokoro-82m_{voice}_{ts}.wav"
        sf.write(str(save_path), audio, sr)

        info = (
            f"Kokoro 82M\n"
            f"Voice: {voice}\n"
            f"Audio: {duration:.1f}s @ {sr}Hz\n"
            f"Latency: {elapsed:.3f}s\n"
            f"RTF: {elapsed/duration:.4f}"
        )
        return str(save_path), info

    except Exception as e:
        return None, f"Kokoro error: {e}"


# ---------------------------------------------------------------------------
# Voxtral — external service
# ---------------------------------------------------------------------------
def check_service(url: str) -> bool:
    for path in ["/v1/models", "/"]:
        try:
            r = httpx.get(f"{url}{path}", timeout=3.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


def generate_voxtral(text: str, voice: str) -> tuple[str | None, str]:
    if not check_service(VOXTRAL_URL):
        return None, "Voxtral not running (start on port 8091)"

    t0 = time.time()
    try:
        response = httpx.post(
            f"{VOXTRAL_URL}/v1/audio/speech",
            json={
                "input": text,
                "model": "mistralai/Voxtral-4B-TTS-2603",
                "response_format": "wav",
                "voice": voice,
            },
            timeout=120.0,
        )
        response.raise_for_status()

        elapsed = time.time() - t0
        audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        duration = len(audio_array) / sr

        ts = int(time.time())
        save_path = OUTPUT_DIR / f"voxtral-4b_{voice}_{ts}.wav"
        sf.write(str(save_path), audio_array, sr)

        info = (
            f"Voxtral 4B TTS\n"
            f"Voice: {voice}\n"
            f"Audio: {duration:.1f}s @ {sr}Hz\n"
            f"Latency: {elapsed:.2f}s\n"
            f"RTF: {elapsed/duration:.3f}"
        )
        return str(save_path), info

    except Exception as e:
        return None, f"Voxtral error: {e}"


# ---------------------------------------------------------------------------
# Fish Audio — external service
# ---------------------------------------------------------------------------
def generate_fish(text: str) -> tuple[str | None, str]:
    if not check_service(FISH_URL):
        return None, "Fish Audio not running (start on port 8092)"

    t0 = time.time()
    try:
        response = httpx.post(
            f"{FISH_URL}/v1/tts",
            json={"text": text, "format": "wav", "streaming": False},
            timeout=120.0,
        )
        response.raise_for_status()

        elapsed = time.time() - t0
        audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        duration = len(audio_array) / sr

        ts = int(time.time())
        save_path = OUTPUT_DIR / f"fish-s2-pro_{ts}.wav"
        sf.write(str(save_path), audio_array, sr)

        info = (
            f"Fish Audio S2 Pro\n"
            f"Audio: {duration:.1f}s @ {sr}Hz\n"
            f"Latency: {elapsed:.2f}s\n"
            f"RTF: {elapsed/duration:.3f}"
        )
        return str(save_path), info

    except Exception as e:
        return None, f"Fish Audio error: {e}"


# ---------------------------------------------------------------------------
# Generate all three
# ---------------------------------------------------------------------------
def generate_all(text: str, voxtral_voice: str, kokoro_voice: str):
    v_audio, v_info = generate_voxtral(text, voxtral_voice)
    f_audio, f_info = generate_fish(text)
    k_audio, k_info = generate_kokoro(text, kokoro_voice)
    return v_audio, v_info, f_audio, f_info, k_audio, k_info


def generate_single(text: str, backend: str, voxtral_voice: str, kokoro_voice: str):
    if backend == "Voxtral 4B":
        return generate_voxtral(text, voxtral_voice)
    elif backend == "Fish Audio S2 Pro":
        return generate_fish(text)
    else:
        return generate_kokoro(text, kokoro_voice)


def check_status():
    voxtral_ok = check_service(VOXTRAL_URL)
    fish_ok = check_service(FISH_URL)
    return "\n".join([
        f"Voxtral ({VOXTRAL_URL}): {'ONLINE' if voxtral_ok else 'OFFLINE'}",
        f"Fish Audio ({FISH_URL}): {'ONLINE' if fish_ok else 'OFFLINE'}",
        f"Kokoro 82M: ALWAYS AVAILABLE (in-process)",
    ])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="TTS Comparison") as demo:
        gr.Markdown(
            "# TTS A/B/C Comparison\n"
            "Voxtral 4B vs Fish Audio S2 Pro vs Kokoro 82M"
        )

        status = gr.Textbox(label="Service Status", interactive=False, lines=3)
        refresh_btn = gr.Button("Refresh Status", size="sm")
        refresh_btn.click(fn=check_status, outputs=[status])
        demo.load(fn=check_status, outputs=[status])

        with gr.Tabs():
            # ---- All Three ----
            with gr.TabItem("Compare All"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_preset = gr.Dropdown(
                            choices=list(SAMPLE_TEXTS.keys()),
                            value="Dialogue", label="Sample Text",
                        )
                        text_input = gr.Textbox(
                            label="Text", value=SAMPLE_TEXTS["Dialogue"], lines=4,
                        )
                        voxtral_voice = gr.Dropdown(
                            choices=VOXTRAL_VOICES, value="casual_male",
                            label="Voxtral Voice",
                        )
                        kokoro_voice = gr.Dropdown(
                            choices=KOKORO_VOICES, value="am_adam",
                            label="Kokoro Voice",
                        )

                        text_preset.change(
                            fn=lambda p: SAMPLE_TEXTS.get(p, ""),
                            inputs=[text_preset], outputs=[text_input],
                        )

                        gen_btn = gr.Button("Generate All", variant="primary", size="lg")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("### Voxtral 4B")
                        v_audio = gr.Audio(label="Voxtral", type="filepath")
                        v_info = gr.Textbox(label="Info", lines=6, interactive=False)
                    with gr.Column():
                        gr.Markdown("### Fish Audio S2 Pro")
                        f_audio = gr.Audio(label="Fish Audio", type="filepath")
                        f_info = gr.Textbox(label="Info", lines=6, interactive=False)
                    with gr.Column():
                        gr.Markdown("### Kokoro 82M")
                        k_audio = gr.Audio(label="Kokoro", type="filepath")
                        k_info = gr.Textbox(label="Info", lines=6, interactive=False)

                gen_btn.click(
                    fn=generate_all,
                    inputs=[text_input, voxtral_voice, kokoro_voice],
                    outputs=[v_audio, v_info, f_audio, f_info, k_audio, k_info],
                )

            # ---- Single Backend ----
            with gr.TabItem("Single"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s_backend = gr.Radio(
                            choices=["Voxtral 4B", "Fish Audio S2 Pro", "Kokoro 82M"],
                            value="Kokoro 82M", label="Backend",
                        )
                        s_text = gr.Textbox(
                            label="Text",
                            value="Welcome, traveler. The road ahead is dangerous.",
                            lines=3,
                        )
                        s_voxtral_voice = gr.Dropdown(
                            choices=VOXTRAL_VOICES, value="casual_male",
                            label="Voxtral Voice",
                        )
                        s_kokoro_voice = gr.Dropdown(
                            choices=KOKORO_VOICES, value="af_heart",
                            label="Kokoro Voice",
                        )
                        s_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=2):
                        s_audio = gr.Audio(label="Output", type="filepath")
                        s_info = gr.Textbox(label="Info", lines=6, interactive=False)

                s_btn.click(
                    fn=generate_single,
                    inputs=[s_text, s_backend, s_voxtral_voice, s_kokoro_voice],
                    outputs=[s_audio, s_info],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="teal"),
        allowed_paths=[str(OUTPUT_DIR)],
    )
