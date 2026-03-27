#!/usr/bin/env python3
"""
Voice Agent — Real-time conversational AI (streaming pipeline)

Pipeline: Silero VAD → Whisper STT → Qwen LLM (streaming) → TTS (chunked) → Speaker

The LLM streams tokens, a sentence chunker detects boundaries, and TTS
synthesizes each chunk immediately — audio starts playing before the LLM
finishes generating.

Run: CUDA_VISIBLE_DEVICES=1 uv run python -u app.py
Requires: vLLM running on :8000 (any Qwen3.5 model)
Optional: Voxtral on :8091 for high-quality TTS
"""

import io
import json
import logging
import os
import re
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
# Sentence Chunker — detects boundaries in streaming token output
# ---------------------------------------------------------------------------
class SentenceChunker:
    """Buffers streaming tokens and yields complete sentences."""

    BOUNDARY = re.compile(r'(?<=[.!?;:])\s+|(?<=[.!?])\s*$')

    def __init__(self, min_first=10, min_rest=30, max_chars=200):
        self.buffer = ""
        self.chunk_count = 0
        self.min_first = min_first   # Aggressive first chunk for fast TTFA
        self.min_rest = min_rest     # Sentence-level for good prosody
        self.max_chars = max_chars   # Force yield on run-on sentences

    @property
    def min_chars(self):
        return self.min_first if self.chunk_count == 0 else self.min_rest

    def feed(self, token: str):
        self.buffer += token

        # Force yield on very long buffers
        if len(self.buffer) >= self.max_chars:
            text = self.buffer.strip()
            if text:
                self.chunk_count += 1
                yield text
            self.buffer = ""
            return

        # First chunk: also split on commas for faster TTFA
        if self.chunk_count == 0:
            pattern = re.compile(r'(?<=[,.!?;:])\s+')
        else:
            pattern = self.BOUNDARY

        matches = list(pattern.finditer(self.buffer))
        if matches:
            last = matches[-1]
            candidate = self.buffer[:last.end()].strip()
            if len(candidate) >= self.min_chars:
                self.chunk_count += 1
                yield candidate
                self.buffer = self.buffer[last.end():]

    def flush(self):
        if self.buffer.strip():
            self.chunk_count += 1
            yield self.buffer.strip()
            self.buffer = ""


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
    pipe = get_kokoro()
    chunks = list(pipe(text, voice="af_heart", speed=1))
    if not chunks:
        return 24000, np.zeros(2400, dtype=np.int16)
    audio = np.concatenate([c[2] for c in chunks])
    return 24000, (audio * 32767).clip(-32768, 32767).astype(np.int16)


def tts_voxtral(text: str) -> tuple[int, np.ndarray] | None:
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
        return sr, (audio * 32767).clip(-32768, 32767).astype(np.int16)
    except Exception as e:
        logger.warning(f"Voxtral TTS failed: {e}")
        return None


def tts_synthesize(text: str, backend: str) -> tuple[int, np.ndarray]:
    """Synthesize with selected backend, fallback to Kokoro."""
    if backend == "voxtral":
        result = tts_voxtral(text)
        if result is not None:
            return result
    return tts_kokoro(text)


# ---------------------------------------------------------------------------
# LLM: Streaming tokens from vLLM
# ---------------------------------------------------------------------------
def stream_llm_tokens(text: str, history: list[dict]):
    """Generator that yields tokens from vLLM streaming endpoint."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": text})

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST",
                f"{LLM_URL}/chat/completions",
                json={
                    "model": "local",
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
    except Exception as e:
        logger.error(f"LLM stream error: {e}")
        yield "Sorry, I couldn't process that."


# ---------------------------------------------------------------------------
# Voice Agent — Streaming Pipeline
# ---------------------------------------------------------------------------
class VoiceAgent:
    def __init__(self):
        self.history: list[dict] = []
        self.tts_backend = "kokoro"
        self.last_latency = {}

    def process_turn_streaming(self, audio_tuple: tuple[int, np.ndarray]):
        """Streaming pipeline: STT → LLM tokens → chunked TTS → yield audio.

        Yields (sample_rate, audio_chunk) tuples as soon as each sentence
        is synthesized — doesn't wait for the full LLM response.
        """
        sr_in, audio_in = audio_tuple
        total_start = time.time()

        # 1. STT
        t0 = time.time()
        stt = get_stt()
        if audio_in.ndim > 1:
            audio_in = audio_in[:, 0] if audio_in.shape[1] < audio_in.shape[0] else audio_in[0]
        if audio_in.dtype != np.float32:
            audio_in = audio_in.astype(np.float32) / max(np.iinfo(audio_in.dtype).max, 1)
        if sr_in != 16000:
            import soxr
            audio_in = soxr.resample(audio_in.reshape(-1), sr_in, 16000)
        result = stt({"raw": audio_in.flatten(), "sampling_rate": 16000})
        user_text = result["text"].strip()
        stt_time = time.time() - t0

        if not user_text:
            return

        logger.info(f"[STT {stt_time:.2f}s] {user_text}")

        # 2. Stream LLM → chunk → TTS → yield audio
        chunker = SentenceChunker()
        full_response = ""
        llm_start = time.time()
        ttfa = None  # Time to first audio
        chunk_count = 0
        tts_total = 0

        for token in stream_llm_tokens(user_text, self.history):
            full_response += token

            for sentence in chunker.feed(token):
                t1 = time.time()
                sr_out, audio_chunk = tts_synthesize(sentence, self.tts_backend)
                tts_time = time.time() - t1
                tts_total += tts_time

                if ttfa is None:
                    ttfa = time.time() - total_start
                    logger.info(f"[TTFA {ttfa:.2f}s] First audio chunk: {sentence!r}")

                chunk_count += 1
                yield sr_out, audio_chunk

        # Flush remaining text
        for sentence in chunker.flush():
            t1 = time.time()
            sr_out, audio_chunk = tts_synthesize(sentence, self.tts_backend)
            tts_total += time.time() - t1
            chunk_count += 1
            yield sr_out, audio_chunk

        llm_time = time.time() - llm_start
        total_time = time.time() - total_start

        if not full_response:
            full_response = "I'm not sure how to respond to that."

        # Update history
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})

        self.last_latency = {
            "stt": stt_time,
            "llm": llm_time,
            "tts": tts_total,
            "ttfa": ttfa or total_time,
            "total": total_time,
            "chunks": chunk_count,
            "user_text": user_text,
            "response_text": full_response,
        }

        logger.info(
            f"[Total {total_time:.2f}s] STT={stt_time:.2f}s LLM={llm_time:.2f}s "
            f"TTS={tts_total:.2f}s TTFA={ttfa:.2f}s chunks={chunk_count}"
        )


agent = VoiceAgent()


def voice_handler(audio: tuple[int, np.ndarray]):
    """FastRTC ReplyOnPause handler — yields audio chunks as they're ready."""
    yield from agent.process_turn_streaming(audio)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Voice Agent") as demo:
        gr.Markdown(
            "# Voice Agent (Streaming)\n"
            "VAD → Whisper → Qwen (streaming) → Kokoro/Voxtral (chunked) → Speaker\n\n"
            "Audio starts playing before the LLM finishes generating."
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
                    f"TTFA: {lat['ttfa']:.2f}s | STT: {lat['stt']:.2f}s | "
                    f"LLM: {lat['llm']:.2f}s | TTS: {lat['tts']:.2f}s"
                )
                lines.append(
                    f"Total: {lat['total']:.2f}s | Chunks: {lat['chunks']}"
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
    auth = os.environ.get("GRADIO_AUTH")
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
