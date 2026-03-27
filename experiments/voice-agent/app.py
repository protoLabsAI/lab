#!/usr/bin/env python3
"""
Voice Agent — Real-time conversational AI (streaming + interrupts + context)

Pipeline: Silero VAD → Whisper STT → Qwen LLM (streaming) → Kokoro TTS (chunked) → Speaker

Features:
  - Streaming LLM→TTS: audio plays before LLM finishes generating
  - Interruption: new speech cancels current response mid-stream
  - Context window: sliding history with summarization for long conversations
  - Pre-warm: all models loaded on startup, zero cold start

Run: CUDA_VISIBLE_DEVICES=1 uv run python -u app.py
Requires: vLLM running on :8000 (any Qwen3.5 model)
"""

import io
import json
import logging
import os
import re
import threading
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

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise — 1-3 sentences max. "
    "Be conversational and natural. Do not use markdown, bullet points, or formatting. "
    "Respond as if speaking out loud."
)

# Context window config
MAX_HISTORY_TURNS = 10  # Keep last 10 turns (20 messages) in full
MAX_HISTORY_TOKENS = 2000  # Rough char estimate (4 chars ≈ 1 token)
SUMMARY_PROMPT = (
    "Summarize this conversation so far in 2-3 sentences. "
    "Focus on key topics discussed and any important facts mentioned."
)


# ---------------------------------------------------------------------------
# Sentence Chunker
# ---------------------------------------------------------------------------
class SentenceChunker:
    BOUNDARY = re.compile(r'(?<=[.!?;:])\s+|(?<=[.!?])\s*$')

    def __init__(self, min_first=10, min_rest=30, max_chars=200):
        self.buffer = ""
        self.chunk_count = 0
        self.min_first = min_first
        self.min_rest = min_rest
        self.max_chars = max_chars

    @property
    def min_chars(self):
        return self.min_first if self.chunk_count == 0 else self.min_rest

    def feed(self, token: str):
        self.buffer += token
        if len(self.buffer) >= self.max_chars:
            text = self.buffer.strip()
            if text:
                self.chunk_count += 1
                yield text
            self.buffer = ""
            return
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
# STT
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
        silence = np.zeros(16000, dtype=np.float32)
        _stt_pipe({"raw": silence, "sampling_rate": 16000})
        logger.info(f"Whisper loaded + warmed in {time.time() - t0:.1f}s")
    return _stt_pipe


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
_kokoro_pipe = None


def get_kokoro():
    global _kokoro_pipe
    if _kokoro_pipe is None:
        logger.info("Loading Kokoro 82M...")
        t0 = time.time()
        from kokoro import KPipeline
        _kokoro_pipe = KPipeline(lang_code="a")
        list(_kokoro_pipe("Hello.", voice="af_heart", speed=1))
        logger.info(f"Kokoro loaded + warmed in {time.time() - t0:.1f}s")
    return _kokoro_pipe


def tts_kokoro(text: str) -> tuple[int, np.ndarray]:
    pipe = get_kokoro()
    chunks = list(pipe(text, voice="af_heart", speed=1))
    if not chunks:
        return 24000, np.zeros(2400, dtype=np.int16)
    audio = np.concatenate([c[2] for c in chunks])
    return 24000, (audio * 32767).clip(-32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# LLM: Streaming with cancellation support
# ---------------------------------------------------------------------------
def stream_llm_tokens(text: str, history: list[dict], cancel: threading.Event):
    """Generator yielding tokens. Stops early if cancel is set."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
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
                    if cancel.is_set():
                        logger.info("[LLM] Cancelled mid-stream")
                        return
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
        if not cancel.is_set():
            logger.error(f"LLM stream error: {e}")
            yield "Sorry, I couldn't process that."


def llm_summarize(history: list[dict]) -> str:
    """Ask the LLM to summarize conversation history."""
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )},
    ]
    try:
        response = httpx.post(
            f"{LLM_URL}/chat/completions",
            json={
                "model": "local",
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=15.0,
        )
        response.raise_for_status()
        msg = response.json()["choices"][0]["message"]
        return (msg.get("content") or "").strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Voice Agent
# ---------------------------------------------------------------------------
class VoiceAgent:
    def __init__(self):
        self.history: list[dict] = []
        self.summary: str = ""  # Rolling summary of older context
        self.cancel = threading.Event()
        self.last_latency = {}
        self._turn_lock = threading.Lock()

    def _get_context_messages(self) -> list[dict]:
        """Build context: summary (if any) + recent history within token budget."""
        messages = []

        # Add summary as system context if we have one
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}",
            })

        # Add recent history, respecting token budget
        recent = self.history[-(MAX_HISTORY_TURNS * 2):]
        total_chars = sum(len(m["content"]) for m in recent)

        # If over budget, trim oldest turns
        while total_chars > MAX_HISTORY_TOKENS and len(recent) > 2:
            removed = recent.pop(0)
            if recent and recent[0]["role"] == "assistant":
                removed2 = recent.pop(0)
                total_chars -= len(removed["content"]) + len(removed2["content"])
            else:
                total_chars -= len(removed["content"])

        messages.extend(recent)
        return messages

    def _maybe_summarize(self):
        """Summarize old history if it's getting long."""
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            # Summarize the oldest turns that are about to be dropped
            old_turns = self.history[:-(MAX_HISTORY_TURNS * 2)]
            if old_turns:
                # Include existing summary for continuity
                to_summarize = []
                if self.summary:
                    to_summarize.append({"role": "system", "content": f"Prior summary: {self.summary}"})
                to_summarize.extend(old_turns)

                new_summary = llm_summarize(to_summarize)
                if new_summary:
                    self.summary = new_summary
                    logger.info(f"[Context] Summarized {len(old_turns)} messages: {new_summary[:80]}...")

                # Trim history to recent window
                self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

    def interrupt(self):
        """Cancel current generation (called when user starts speaking)."""
        self.cancel.set()

    def process_turn_streaming(self, audio_tuple: tuple[int, np.ndarray]):
        """Full streaming pipeline with interrupt support."""
        # Reset cancellation for this turn
        self.cancel.clear()

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
        ttfa = None
        chunk_count = 0
        tts_total = 0
        interrupted = False

        context = self._get_context_messages()

        for token in stream_llm_tokens(user_text, context, self.cancel):
            if self.cancel.is_set():
                interrupted = True
                break

            full_response += token

            for sentence in chunker.feed(token):
                if self.cancel.is_set():
                    interrupted = True
                    break

                t1 = time.time()
                sr_out, audio_chunk = tts_kokoro(sentence)
                tts_total += time.time() - t1

                if ttfa is None:
                    ttfa = time.time() - total_start
                    logger.info(f"[TTFA {ttfa:.2f}s] First chunk: {sentence!r}")

                chunk_count += 1
                yield sr_out, audio_chunk

            if interrupted:
                break

        # Flush remaining (unless interrupted)
        if not interrupted:
            for sentence in chunker.flush():
                if self.cancel.is_set():
                    break
                t1 = time.time()
                sr_out, audio_chunk = tts_kokoro(sentence)
                tts_total += time.time() - t1
                chunk_count += 1
                yield sr_out, audio_chunk

        llm_time = time.time() - llm_start
        total_time = time.time() - total_start

        # Update history (even partial responses are useful context)
        if full_response.strip():
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": full_response.strip()})
            self._maybe_summarize()

        status = "interrupted" if interrupted else "complete"
        self.last_latency = {
            "stt": stt_time, "llm": llm_time, "tts": tts_total,
            "ttfa": ttfa or total_time, "total": total_time,
            "chunks": chunk_count, "status": status,
            "history_len": len(self.history),
            "has_summary": bool(self.summary),
        }

        logger.info(
            f"[{status.upper()} {total_time:.2f}s] STT={stt_time:.2f}s LLM={llm_time:.2f}s "
            f"TTS={tts_total:.2f}s TTFA={ttfa or 0:.2f}s chunks={chunk_count} "
            f"history={len(self.history)} summary={'yes' if self.summary else 'no'}"
        )


agent = VoiceAgent()


def prewarm():
    logger.info("Pre-warming all models...")
    t0 = time.time()
    get_stt()
    get_kokoro()
    try:
        httpx.post(
            f"{LLM_URL}/chat/completions",
            json={
                "model": "local",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 1,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30.0,
        )
        logger.info("LLM prefix cache warmed")
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")
    logger.info(f"All models pre-warmed in {time.time() - t0:.1f}s")


def voice_handler(audio: tuple[int, np.ndarray]):
    # Signal interruption to any in-progress generation
    agent.interrupt()
    yield from agent.process_turn_streaming(audio)


def build_ui():
    with gr.Blocks(title="Voice Agent") as demo:
        gr.Markdown("# Voice Agent\nSpeak — I'll respond.")
        from fastrtc.reply_on_pause import AlgoOptions
        Stream(
            ReplyOnPause(
                voice_handler,
                algo_options=AlgoOptions(
                    audio_chunk_duration=0.6,
                    started_talking_threshold=0.5,
                    speech_threshold=0.1,
                ),
                output_sample_rate=24000,
                can_interrupt=True,
            ),
            modality="audio",
            mode="send-receive",
        )
    return demo


if __name__ == "__main__":
    prewarm()
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
