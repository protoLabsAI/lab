#!/usr/bin/env python3
"""
TTS A/B/C Comparison — Voxtral 4B vs Fish Audio S2 Pro vs Kokoro 82M

Voxtral + Fish Audio run as external services (GPU-heavy).
Kokoro runs in-process (82M params, ~2GB VRAM, trivial).

Run: uv run python -u app.py
"""

import base64
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

# Fish S2-Pro inline control tags (supports 15k+ tags; these are the most-used)
# Insert into text to control prosody/emotion/delivery.
CLONE_PROMPTS = {
    "Neutral intro": "Hi, my name is Morgan. It's nice to meet you.",
    "Emotional range": "[excited] We finally did it! [pause] [whisper] I never thought this day would come.",
    "Laughter": "That was the funniest thing I've ever heard. [laughing] Oh my gosh, I can't even.",
    "Broadcast": "[professional broadcast tone] In today's top story, researchers announced a breakthrough in voice synthesis.",
    "Whisper + emphasis": "[whisper] Listen carefully. The treasure is hidden [emphasis] behind the third stone from the left.",
    "Dialogue": "I told him no. [pause] He just didn't want to hear it. [sighs] What was I supposed to do?",
    "Technical read": "The transformer architecture uses self-attention mechanisms to process input sequences in parallel.",
}

CONTROL_TAGS = [
    "[pause]", "[emphasis]", "[laughing]", "[inhale]", "[sighs]",
    "[whisper]", "[excited]", "[angry]", "[sad]", "[singing]",
    "[pitch up]", "[pitch down]", "[volume up]", "[volume down]",
    "[professional broadcast tone]", "[whisper in small voice]",
]

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
# Fish Audio — Voice Cloning (references API)
# ---------------------------------------------------------------------------
def _file_to_b64(path: str) -> str:
    """Read audio file and return base64 string (Fish decodes len>255 as b64)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def generate_fish_clone(
    text: str,
    reference_audio: str | None,
    reference_text: str,
    reference_id: str,
    seed: int | None,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_length: int,
    use_memory_cache: bool,
) -> tuple[str | None, str]:
    """Fish Audio voice cloning — either inline reference or saved reference_id."""
    if not check_service(FISH_URL):
        return None, "Fish Audio not running (start on port 8092)"

    if not text or not text.strip():
        return None, "Text is required"

    # Build payload — prefer saved reference_id, else require inline ref + transcript
    payload: dict = {
        "text": text,
        "format": "wav",
        "streaming": False,
        "seed": seed if seed and seed > 0 else None,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "chunk_length": int(chunk_length),
        "use_memory_cache": "on" if use_memory_cache else "off",
    }

    ref_id = (reference_id or "").strip()
    if ref_id:
        payload["reference_id"] = ref_id
        mode = f"saved id='{ref_id}'"
    else:
        if not reference_audio:
            return None, "Provide reference audio (upload/record) or a saved reference ID"
        if not reference_text or not reference_text.strip():
            return None, "Reference text (transcript of the audio) is required for inline cloning"
        payload["references"] = [
            {"audio": _file_to_b64(reference_audio), "text": reference_text.strip()}
        ]
        mode = f"inline ref ({Path(reference_audio).name})"

    t0 = time.time()
    try:
        response = httpx.post(
            f"{FISH_URL}/v1/tts",
            json=payload,
            headers={"authorization": "Bearer none"},
            timeout=300.0,
        )
        response.raise_for_status()

        elapsed = time.time() - t0
        audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        duration = len(audio_array) / sr

        ts = int(time.time())
        tag = ref_id if ref_id else "clone"
        save_path = OUTPUT_DIR / f"fish-s2-pro_{tag}_{ts}.wav"
        sf.write(str(save_path), audio_array, sr)

        info = (
            f"Fish Audio S2 Pro (voice clone)\n"
            f"Mode: {mode}\n"
            f"Audio: {duration:.1f}s @ {sr}Hz\n"
            f"Latency: {elapsed:.2f}s\n"
            f"RTF: {elapsed/duration:.3f}\n"
            f"Settings: T={temperature} top_p={top_p} seed={seed} cache={'on' if use_memory_cache else 'off'}"
        )
        return str(save_path), info

    except httpx.HTTPStatusError as e:
        return None, f"Fish Audio HTTP {e.response.status_code}: {e.response.text[:500]}"
    except Exception as e:
        return None, f"Fish Audio clone error: {e}"


def fish_list_references() -> list[str]:
    """Fetch saved reference IDs from Fish server."""
    if not check_service(FISH_URL):
        return []
    try:
        r = httpx.get(f"{FISH_URL}/v1/references/list", timeout=5.0)
        r.raise_for_status()
        return r.json().get("reference_ids", [])
    except Exception as e:
        logger.warning(f"Failed to list refs: {e}")
        return []


def fish_add_reference(
    ref_id: str, reference_audio: str | None, reference_text: str
) -> tuple[str, list[str]]:
    """Save a named reference voice to Fish server. Returns (status, refreshed list)."""
    if not check_service(FISH_URL):
        return "Fish Audio not running", []

    ref_id = (ref_id or "").strip()
    if not ref_id:
        return "Reference ID is required", fish_list_references()
    if not reference_audio:
        return "Reference audio file is required", fish_list_references()
    if not reference_text or not reference_text.strip():
        return "Reference text (transcript) is required", fish_list_references()

    try:
        with open(reference_audio, "rb") as f:
            files = {"audio": (Path(reference_audio).name, f.read(), "audio/wav")}
        data = {"id": ref_id, "text": reference_text.strip()}
        r = httpx.post(
            f"{FISH_URL}/v1/references/add",
            data=data,
            files=files,
            timeout=30.0,
        )
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        msg = body.get("message", r.text[:200])
        status = f"[{r.status_code}] {msg}"
        return status, fish_list_references()
    except Exception as e:
        return f"Add failed: {e}", fish_list_references()


def fish_delete_reference(ref_id: str) -> tuple[str, list[str]]:
    if not ref_id:
        return "Pick a reference to delete", fish_list_references()
    try:
        r = httpx.request(
            "DELETE",
            f"{FISH_URL}/v1/references/delete",
            json={"reference_id": ref_id},
            timeout=10.0,
        )
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return f"[{r.status_code}] {body.get('message', r.text[:200])}", fish_list_references()
    except Exception as e:
        return f"Delete failed: {e}", fish_list_references()


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
    with gr.Blocks(title="TTS Comparison", theme=gr.themes.Soft(primary_hue="teal")) as demo:
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

            # ---- Voice Cloning (Fish S2 Pro) ----
            with gr.TabItem("Voice Clone (Fish)"):
                gr.Markdown(
                    "### Fish Audio S2 Pro — Voice Cloning\n"
                    "Clone a voice from 10–30s of reference audio + its transcript. "
                    "Use inline `[tags]` in the target text for prosody/emotion control."
                )

                with gr.Row():
                    # LEFT: Target text + controls
                    with gr.Column(scale=2):
                        clone_preset = gr.Dropdown(
                            choices=list(CLONE_PROMPTS.keys()),
                            value="Emotional range",
                            label="Example Prompt (with control tags)",
                        )
                        clone_text = gr.Textbox(
                            label="Target Text (what the cloned voice will say)",
                            value=CLONE_PROMPTS["Emotional range"],
                            lines=4,
                        )
                        clone_preset.change(
                            fn=lambda p: CLONE_PROMPTS.get(p, ""),
                            inputs=[clone_preset], outputs=[clone_text],
                        )

                        with gr.Accordion("Insert control tag", open=False):
                            tag_picker = gr.Dropdown(
                                choices=CONTROL_TAGS, value="[pause]",
                                label="Tag", scale=1,
                            )
                            tag_insert_btn = gr.Button("Append tag to text", size="sm")
                            tag_insert_btn.click(
                                fn=lambda t, tag: (t or "") + (" " if t and not t.endswith(" ") else "") + tag,
                                inputs=[clone_text, tag_picker],
                                outputs=[clone_text],
                            )

                        with gr.Accordion("Advanced sampling", open=False):
                            with gr.Row():
                                clone_temp = gr.Slider(
                                    0.1, 1.0, value=0.8, step=0.05,
                                    label="Temperature",
                                )
                                clone_top_p = gr.Slider(
                                    0.1, 1.0, value=0.8, step=0.05,
                                    label="Top-p",
                                )
                            with gr.Row():
                                clone_rep_pen = gr.Slider(
                                    0.9, 2.0, value=1.1, step=0.05,
                                    label="Repetition penalty",
                                )
                                clone_chunk = gr.Slider(
                                    100, 300, value=300, step=10,
                                    label="Chunk length",
                                )
                            with gr.Row():
                                clone_seed = gr.Number(
                                    value=0, precision=0, label="Seed (0=random)",
                                )
                                clone_cache = gr.Checkbox(
                                    value=True,
                                    label="Memory cache (reuse encoded ref)",
                                )

                        clone_gen_btn = gr.Button(
                            "Generate Cloned Speech", variant="primary", size="lg",
                        )

                    # RIGHT: Reference source (inline upload OR saved id)
                    with gr.Column(scale=1):
                        gr.Markdown("#### Reference Voice")

                        with gr.Tabs():
                            with gr.TabItem("Inline (upload)"):
                                clone_ref_audio = gr.Audio(
                                    label="Reference audio (10–30s, WAV/MP3)",
                                    type="filepath",
                                    sources=["upload", "microphone"],
                                )
                                clone_ref_text = gr.Textbox(
                                    label="Reference transcript (what is said in the audio)",
                                    lines=3,
                                    placeholder="Exact words spoken in the reference clip",
                                )

                            with gr.TabItem("Saved reference"):
                                saved_refs = gr.Dropdown(
                                    choices=[], value=None,
                                    label="Saved reference ID", allow_custom_value=True,
                                )
                                with gr.Row():
                                    refresh_refs_btn = gr.Button("Refresh", size="sm")
                                    delete_ref_btn = gr.Button(
                                        "Delete selected", size="sm", variant="stop",
                                    )
                                refs_status = gr.Textbox(
                                    label="Status", interactive=False, lines=1,
                                )

                            with gr.TabItem("Save new reference"):
                                save_id = gr.Textbox(
                                    label="Reference ID (alphanumeric, - _ space)",
                                    placeholder="e.g. morgan_freeman_sample1",
                                )
                                save_audio = gr.Audio(
                                    label="Reference audio (10–30s)",
                                    type="filepath",
                                    sources=["upload", "microphone"],
                                )
                                save_text = gr.Textbox(
                                    label="Reference transcript", lines=3,
                                )
                                save_btn = gr.Button(
                                    "Save to server", variant="primary", size="sm",
                                )
                                save_status = gr.Textbox(
                                    label="Status", interactive=False, lines=1,
                                )

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Cloned Output", type="filepath")
                    clone_info = gr.Textbox(label="Info", lines=8, interactive=False)

                # Wire up cloning
                def _clone_entrypoint(
                    text, inline_audio, inline_text, saved_id,
                    seed, temp, top_p, rep_pen, chunk, cache,
                ):
                    # Saved id takes priority if provided
                    ref_id = (saved_id or "").strip()
                    seed_i = int(seed) if seed and int(seed) > 0 else None
                    return generate_fish_clone(
                        text=text,
                        reference_audio=inline_audio if not ref_id else None,
                        reference_text=inline_text if not ref_id else "",
                        reference_id=ref_id,
                        seed=seed_i,
                        temperature=temp,
                        top_p=top_p,
                        repetition_penalty=rep_pen,
                        chunk_length=chunk,
                        use_memory_cache=bool(cache),
                    )

                clone_gen_btn.click(
                    fn=_clone_entrypoint,
                    inputs=[
                        clone_text, clone_ref_audio, clone_ref_text, saved_refs,
                        clone_seed, clone_temp, clone_top_p, clone_rep_pen,
                        clone_chunk, clone_cache,
                    ],
                    outputs=[clone_audio_out, clone_info],
                )

                # Refresh saved refs on tab load + button
                def _refresh_refs():
                    ids = fish_list_references()
                    return gr.update(choices=ids, value=(ids[0] if ids else None)), f"{len(ids)} saved"
                refresh_refs_btn.click(fn=_refresh_refs, outputs=[saved_refs, refs_status])
                demo.load(fn=_refresh_refs, outputs=[saved_refs, refs_status])

                # Save new reference
                def _save_ref(rid, audio, text):
                    status, ids = fish_add_reference(rid, audio, text)
                    return status, gr.update(choices=ids)
                save_btn.click(
                    fn=_save_ref,
                    inputs=[save_id, save_audio, save_text],
                    outputs=[save_status, saved_refs],
                )

                # Delete
                def _del_ref(rid):
                    status, ids = fish_delete_reference(rid)
                    return status, gr.update(choices=ids, value=(ids[0] if ids else None))
                delete_ref_btn.click(
                    fn=_del_ref, inputs=[saved_refs],
                    outputs=[refs_status, saved_refs],
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
        allowed_paths=[str(OUTPUT_DIR)],
    )
