#!/usr/bin/env python3
"""
Voice Agent — Real-time conversational AI

Pipeline: Silero VAD → Whisper STT → Qwen LLM → TTS → Speaker

Three TTS backends (swap in UI):
  - Kokoro 82M: ~50ms latency, in-process, Apache 2.0
  - Voxtral 4B: ~250ms latency, external service, voice cloning, Qwen encoder
  - None: text-only (debug mode)

STT: whisper-large-v3-turbo via HuggingFace Transformers (~120ms for 3-5s utterance)
LLM: Qwen3.5 via vLLM on localhost:8000 (streaming tokens)
VAD: Silero VAD via FastRTC ReplyOnPause

Run: CUDA_VISIBLE_DEVICES=1 uv run python -u app.py
Requires: vLLM running on :8000 (any Qwen3.5 model)
Optional: Voxtral on :8091 for high-quality TTS
"""

import io
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
import torch
from fastrtc import ReplyOnPause, Stream
from transformers import pipeline as hf_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda"
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1")
VOXTRAL_URL = os.environ.get("VOXTRAL_URL", "http://localhost:8091")
VOXTRAL_VOICE = os.environ.get("VOXTRAL_VOICE", "casual_male")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise — 1-3 sentences max. "
    "Be conversational and natural. Do not use markdown, bullet points, or formatting. "
    "Respond as if speaking out loud."
)

# ---------------------------------------------------------------------------
# STT: Whisper large-v3-turbo
# ---------------------------------------------------------------------------
_stt_pipe = None


def get_stt():
    global _stt_pipe
    if _stt_pipe is None:
        logger.info("Loading Whisper large-v3-turbo...")
        t0 = time.time()
        _stt_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=DEVICE,
            model_kwargs={"attn_implementation": "sdpa"},
        )
        logger.info(f"Whisper loaded in {time.time() - t0:.1f}s")
    return _stt_pipe


# ---------------------------------------------------------------------------
# TTS: Kokoro (in-process) or Voxtral (external)
# ---------------------------------------------------------------------------
_kokoro_pipe = None


def get_kokoro():
    global _kokoro_pipe
    if _kokoro_pipe is None:
        logger.info("Loading Kokoro 82M...")
        from kokoro import KPipeline
        _kokoro_pipe = KPipeline(lang_code="a")
        logger.info("Kokoro loaded")
    return _kokoro_pipe


def tts_kokoro(text: str) -> tuple[int, np.ndarray]:
    """Synthesize speech with Kokoro. Returns (sample_rate, int16_audio)."""
    pipe = get_kokoro()
    chunks = list(pipe(text, voice="af_heart", speed=1))
    if not chunks:
        return 24000, np.zeros(2400, dtype=np.int16)
    audio = np.concatenate([c[2] for c in chunks])
    # Convert float32 [-1,1] to int16 for FastRTC
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return 24000, audio_int16


def tts_voxtral(text: str) -> tuple[int, np.ndarray] | None:
    """Synthesize speech with Voxtral. Returns (sample_rate, audio_array) or None."""
    try:
        response = httpx.post(
            f"{VOXTRAL_URL}/v1/audio/speech",
            json={
                "input": text,
                "model": "mistralai/Voxtral-4B-TTS-2603",
                "response_format": "wav",
                "voice": VOXTRAL_VOICE,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        # Convert to int16 for FastRTC
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        return sr, audio_int16
    except Exception as e:
        logger.warning(f"Voxtral TTS failed: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM: Qwen via vLLM (streaming)
# ---------------------------------------------------------------------------
def llm_respond(text: str, history: list[dict]) -> str:
    """Get LLM response via vLLM OpenAI-compatible API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])  # Keep last 3 turns
    messages.append({"role": "user", "content": text})

    try:
        response = httpx.post(
            f"{LLM_URL}/chat/completions",
            json={
                "model": "local",
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        msg = response.json()["choices"][0]["message"]
        # Content may be null if model used reasoning — check reasoning field too
        content = msg.get("content") or ""
        if not content and msg.get("reasoning"):
            content = msg["reasoning"]
        return content.strip() if content else "I'm not sure how to respond to that."
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Sorry, I couldn't process that."


# ---------------------------------------------------------------------------
# Voice Agent Pipeline
# ---------------------------------------------------------------------------
class VoiceAgent:
    def __init__(self):
        self.history: list[dict] = []
        self.tts_backend = "kokoro"
        self.last_latency = {}

    def process_turn(self, audio_tuple: tuple[int, np.ndarray]) -> tuple[int, np.ndarray] | None:
        """Full pipeline: audio in → audio out."""
        sr_in, audio_in = audio_tuple
        total_start = time.time()

        # 1. STT
        t0 = time.time()
        stt = get_stt()
        # Ensure 1D mono
        if audio_in.ndim > 1:
            audio_in = audio_in[:, 0] if audio_in.shape[1] < audio_in.shape[0] else audio_in[0]
        # Convert to float32 for Whisper
        if audio_in.dtype != np.float32:
            audio_in = audio_in.astype(np.float32) / max(np.iinfo(audio_in.dtype).max, 1)
        # Resample to 16kHz if needed
        if sr_in != 16000:
            import soxr
            audio_in = soxr.resample(audio_in.reshape(-1), sr_in, 16000)
        result = stt({"raw": audio_in.flatten(), "sampling_rate": 16000})
        user_text = result["text"].strip()
        stt_time = time.time() - t0

        if not user_text:
            return None

        logger.info(f"[STT {stt_time:.2f}s] {user_text}")

        # 2. LLM
        t0 = time.time()
        response_text = llm_respond(user_text, self.history)
        llm_time = time.time() - t0
        if not response_text:
            response_text = "I'm not sure how to respond to that."
        logger.info(f"[LLM {llm_time:.2f}s] {response_text}")

        # Update history
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response_text})

        # 3. TTS
        t0 = time.time()
        if self.tts_backend == "voxtral":
            tts_result = tts_voxtral(response_text)
        else:
            tts_result = tts_kokoro(response_text)
        tts_time = time.time() - t0

        if tts_result is None:
            # Voxtral failed, fall back to kokoro
            t0 = time.time()
            tts_result = tts_kokoro(response_text)
            tts_time = time.time() - t0

        total_time = time.time() - total_start
        logger.info(
            f"[Total {total_time:.2f}s] STT={stt_time:.2f}s LLM={llm_time:.2f}s TTS={tts_time:.2f}s"
        )

        self.last_latency = {
            "stt": stt_time,
            "llm": llm_time,
            "tts": tts_time,
            "total": total_time,
            "user_text": user_text,
            "response_text": response_text,
        }

        sr_out, audio_out = tts_result
        # FastRTC expects (sample_rate, audio_array)
        return sr_out, audio_out


agent = VoiceAgent()


def voice_handler(audio: tuple[int, np.ndarray]):
    """FastRTC ReplyOnPause handler — called when user stops speaking."""
    result = agent.process_turn(audio)
    if result is not None:
        yield result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Voice Agent") as demo:
        gr.Markdown(
            "# Voice Agent\n"
            "Silero VAD → Whisper Turbo → Qwen LLM → Kokoro/Voxtral TTS\n\n"
            "Speak into your mic — the agent responds when you pause."
        )

        def get_log():
            lat = agent.last_latency
            if not lat:
                return "Waiting for first turn..."
            lines = []
            for i in range(0, len(agent.history), 2):
                if i + 1 < len(agent.history):
                    lines.append(f"YOU: {agent.history[i]['content']}")
                    lines.append(f"AI:  {agent.history[i+1]['content']}")
                    lines.append("")
            if lat:
                lines.append("--- Last Turn ---")
                lines.append(
                    f"STT: {lat['stt']:.2f}s | LLM: {lat['llm']:.2f}s | "
                    f"TTS: {lat['tts']:.2f}s | Total: {lat['total']:.2f}s"
                )
            return "\n".join(lines)

        def set_tts(v):
            agent.tts_backend = v
            return f"TTS set to {v}"

        def clear_history():
            agent.history.clear()
            agent.last_latency = {}
            return "History cleared."

        with gr.Row():
            with gr.Column(scale=1):
                tts_select = gr.Radio(
                    choices=["kokoro", "voxtral"],
                    value="kokoro",
                    label="TTS Backend",
                    info="Kokoro: ~50ms, in-process | Voxtral: ~250ms, better quality",
                )
                tts_status = gr.Textbox(label="", lines=1, interactive=False)
                tts_select.change(fn=set_tts, inputs=[tts_select], outputs=[tts_status])

                gr.Markdown("### Conversation Log")
                log_box = gr.Textbox(label="Log", lines=15, interactive=False)
                refresh_btn = gr.Button("Refresh Log", variant="secondary")
                refresh_btn.click(fn=get_log, outputs=[log_box])

                clear_btn = gr.Button("Clear History", size="sm")
                clear_btn.click(fn=clear_history, outputs=[log_box])

            with gr.Column(scale=2):
                from fastrtc.reply_on_pause import AlgoOptions
                webrtc = Stream(
                    ReplyOnPause(
                        voice_handler,
                        algo_options=AlgoOptions(
                            audio_chunk_duration=0.6,
                            started_talking_threshold=0.5,
                            speech_threshold=0.1,
                        ),
                        output_sample_rate=24000,
                    ),
                    modality="audio",
                    mode="send-receive",
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # For mic access over Tailscale, use: sudo tailscale funnel 7866
    # Then access via https://protolabs.taild25506.ts.net:443
    auth = os.environ.get("GRADIO_AUTH")  # format: "user:pass" or "user1:pass1,user2:pass2"
    auth_pairs = None
    if auth:
        auth_pairs = [tuple(pair.split(":")) for pair in auth.split(",")]
        logger.info(f"Auth enabled for {len(auth_pairs)} user(s)")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False,
        show_error=True,
        auth=auth_pairs,
    )
