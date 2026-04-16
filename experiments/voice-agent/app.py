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

# TTS backend selection
# - "kokoro": in-process Kokoro 82M (24kHz, ~95ms/chunk, no cloning)
# - "fish":   external Fish S2 Pro on FISH_URL (44.1kHz, streaming, voice cloning)
TTS_BACKEND = os.environ.get("TTS_BACKEND", "kokoro").lower()
FISH_URL = os.environ.get("FISH_URL", "http://localhost:8092")
FISH_REFERENCE_ID = os.environ.get("FISH_REFERENCE_ID", "").strip() or None
FISH_SR = 44100
KOKORO_SR = 24000
TTS_SAMPLE_RATE = FISH_SR if TTS_BACKEND == "fish" else KOKORO_SR

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
        return KOKORO_SR, np.zeros(2400, dtype=np.int16)
    audio = np.concatenate([c[2] for c in chunks])
    return KOKORO_SR, (audio * 32767).clip(-32768, 32767).astype(np.int16)


def tts_kokoro_stream(text: str, cancel: threading.Event | None = None):
    """Wrap Kokoro as a single-yield generator for uniform pipeline."""
    sr, audio = tts_kokoro(text)
    yield sr, audio


# Currently selected Fish voice (mutable; UI updates this via the dropdown).
# None = Fish's baked-in default voice.
_current_fish_voice: str | None = FISH_REFERENCE_ID


def set_fish_voice(voice_id: str | None):
    """Swap the active Fish voice cloning reference."""
    global _current_fish_voice
    _current_fish_voice = (voice_id or "").strip() or None
    logger.info(f"[voice] Switched Fish reference to {_current_fish_voice!r}")
    return _current_fish_voice


def fish_list_voices() -> list[str]:
    """Query saved reference voice IDs from the Fish server."""
    try:
        r = httpx.get(
            f"{FISH_URL}/v1/references/list",
            headers={"Accept": "application/json"},
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json().get("reference_ids", [])
    except Exception as e:
        logger.warning(f"Failed to list Fish voices: {e}")
        return []


def fish_save_voice(audio_path: str, transcript: str, voice_id: str) -> tuple[bool, str]:
    """Save an audio clip + transcript to the Fish server as a named reference."""
    if not audio_path:
        return False, "Upload or record audio first."
    voice_id = (voice_id or "").strip()
    if not voice_id:
        return False, "Voice ID is required."
    import re as _re
    if not _re.match(r"^[a-zA-Z0-9\-_ ]+$", voice_id) or len(voice_id) > 255:
        return False, "Voice ID: alphanumeric, '-', '_', space only (max 255)."
    transcript = (transcript or "").strip()
    if not transcript:
        return False, "Transcript is required (or click 'Auto-transcribe' first)."

    try:
        with open(audio_path, "rb") as f:
            name = Path(audio_path).name
            files = {"audio": (name, f.read(), "application/octet-stream")}
        r = httpx.post(
            f"{FISH_URL}/v1/references/add",
            data={"id": voice_id, "text": transcript},
            files=files,
            headers={"Accept": "application/json"},
            timeout=30.0,
        )
        if r.status_code == 200:
            return True, f"Saved '{voice_id}'."
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return False, f"[{r.status_code}] {body.get('message', r.text[:160])}"
    except Exception as e:
        return False, f"Save failed: {e}"


def whisper_transcribe_file(audio_path: str) -> str:
    """Use the already-loaded Whisper model to transcribe a local audio file.
    Passes return_timestamps=True so clips >30s use long-form chunking."""
    if not audio_path:
        return ""
    try:
        stt = get_stt()
        result = stt(audio_path, return_timestamps=True)
        return (result.get("text") or "").strip()
    except Exception as e:
        logger.warning(f"Whisper transcribe failed: {e}")
        return ""


def tts_fish_stream(text: str, cancel: threading.Event | None = None):
    """Stream Fish Audio S2 Pro chunks as int16 PCM @ 44.1kHz.

    Wire format: ~44 byte WAV header, then raw int16 LE PCM samples (mono).
    See fish-speech/tools/server/inference.py:30-33.
    """
    payload = {
        "text": text,
        "format": "wav",
        "streaming": True,
        "use_memory_cache": "on",  # reuse encoded ref between sentences
    }
    if _current_fish_voice:
        payload["reference_id"] = _current_fish_voice

    buf = bytearray()
    header_skipped = False
    HEADER_BYTES = 44

    try:
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{FISH_URL}/v1/tts",
                json=payload,
                headers={"authorization": "Bearer none"},
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    if cancel is not None and cancel.is_set():
                        return
                    if not chunk:
                        continue
                    buf.extend(chunk)

                    # Skip 44-byte WAV header on first valid prefix
                    if not header_skipped:
                        if len(buf) < HEADER_BYTES:
                            continue
                        del buf[:HEADER_BYTES]
                        header_skipped = True

                    # Yield aligned int16 samples; carry remainder for next round
                    n_samples = len(buf) // 2
                    if n_samples == 0:
                        continue
                    usable = n_samples * 2
                    arr = np.frombuffer(bytes(buf[:usable]), dtype=np.int16)
                    del buf[:usable]
                    yield FISH_SR, arr
    except Exception as e:
        logger.error(f"Fish TTS stream error: {e}")
        # Yield a brief silence so the pipeline doesn't stall silently
        yield FISH_SR, np.zeros(int(FISH_SR * 0.2), dtype=np.int16)


def tts_stream(text: str, cancel: threading.Event | None = None):
    """Dispatch to the configured TTS backend as a (sr, int16_array) generator."""
    if TTS_BACKEND == "fish":
        yield from tts_fish_stream(text, cancel=cancel)
    else:
        yield from tts_kokoro_stream(text, cancel=cancel)


# ---------------------------------------------------------------------------
# LLM: Streaming with cancellation support
# ---------------------------------------------------------------------------
def stream_llm_tokens(text: str, history: list[dict], cancel: threading.Event):
    """Generator yielding tokens. Stops early if cancel is set.

    Qwen3.5 chat template rejects multiple system messages — merge the base
    SYSTEM_PROMPT with any history-injected summary into a single system
    message at position 0, then append non-system history + current user turn.
    """
    system_parts = [SYSTEM_PROMPT]
    non_system_history = []
    for m in history:
        if m.get("role") == "system":
            system_parts.append(m["content"])
        else:
            non_system_history.append(m)
    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    messages.extend(non_system_history)
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
                for sr_out, audio_chunk in tts_stream(sentence, cancel=self.cancel):
                    if self.cancel.is_set():
                        interrupted = True
                        break
                    if ttfa is None:
                        ttfa = time.time() - total_start
                        logger.info(f"[TTFA {ttfa:.2f}s] First chunk: {sentence!r}")
                    chunk_count += 1
                    yield sr_out, audio_chunk
                tts_total += time.time() - t1

            if interrupted:
                break

        # Flush remaining (unless interrupted)
        if not interrupted:
            for sentence in chunker.flush():
                if self.cancel.is_set():
                    break
                t1 = time.time()
                for sr_out, audio_chunk in tts_stream(sentence, cancel=self.cancel):
                    if self.cancel.is_set():
                        break
                    chunk_count += 1
                    yield sr_out, audio_chunk
                tts_total += time.time() - t1

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
    logger.info(
        f"Pre-warming all models... (TTS_BACKEND={TTS_BACKEND}, sr={TTS_SAMPLE_RATE})"
    )
    t0 = time.time()
    get_stt()

    # Always preload Kokoro (cheap fallback) — even if Fish is selected, it's
    # available if Fish goes down mid-session.
    get_kokoro()

    # Fish health check + first-call warmup (torch.compile is ~2 min cold).
    if TTS_BACKEND == "fish":
        try:
            r = httpx.get(f"{FISH_URL}/v1/health", timeout=5.0)
            r.raise_for_status()
            logger.info(f"Fish reachable at {FISH_URL}")
            if FISH_REFERENCE_ID:
                logger.info(f"Fish voice clone reference: {FISH_REFERENCE_ID!r}")
        except Exception as e:
            logger.error(f"Fish unreachable at {FISH_URL}: {e}")
            logger.error("Start it with: cd ~/dev/fish-speech && CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.api_server --listen 0.0.0.0:8092 --llama-checkpoint-path checkpoints/s2-pro --decoder-checkpoint-path checkpoints/s2-pro/codec.pth --decoder-config-name modded_dac_vq --half --compile")

        try:
            t_fish = time.time()
            # Drain a tiny stream to amortize compile + cache the model
            chunks = 0
            for _ in tts_fish_stream("Ready."):
                chunks += 1
            logger.info(f"Fish warmed in {time.time() - t_fish:.1f}s ({chunks} chunks)")
        except Exception as e:
            logger.warning(f"Fish warmup failed: {e}")

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


def voice_handler(audio: tuple[int, np.ndarray], *extra_inputs):
    """Main handler. If TTS_BACKEND=fish, the last extra input is the voice-dropdown
    value from FastRTC additional_inputs. (FastRTC prepends a WebRTC session id
    as the first extra, so we always take the LAST non-empty string.)"""
    if TTS_BACKEND == "fish":
        # Pick the last string extra (the voice dropdown value) and ignore the
        # WebRTC session id that FastRTC prepends.
        v = None
        for item in reversed(extra_inputs):
            if isinstance(item, str) and item:
                v = item
                break
        set_fish_voice(None if (not v or v == "(default)") else v)

    # Signal interruption to any in-progress generation
    agent.interrupt()
    yield from agent.process_turn_streaming(audio)


def build_stream():
    """Build the FastRTC Stream with the Fish voice selector as an extra input.
    When Fish backend is active, also mounts an "Add voice clone" accordion into
    the generated Blocks so new references can be recorded without leaving the
    voice-agent UI.
    """
    from fastrtc.reply_on_pause import AlgoOptions

    additional_inputs = []
    voice_dd = None
    if TTS_BACKEND == "fish":
        voices = fish_list_voices()
        default_choice = (
            _current_fish_voice if _current_fish_voice in voices else "(default)"
        )
        voice_dd = gr.Dropdown(
            choices=["(default)"] + voices,
            value=default_choice,
            label="Fish voice (select a saved reference)",
            allow_custom_value=True,
        )
        additional_inputs.append(voice_dd)

    stream = Stream(
        ReplyOnPause(
            voice_handler,
            algo_options=AlgoOptions(
                audio_chunk_duration=0.6,
                started_talking_threshold=0.5,
                speech_threshold=0.1,
            ),
            output_sample_rate=TTS_SAMPLE_RATE,
            can_interrupt=True,
        ),
        additional_inputs=additional_inputs,
        modality="audio",
        mode="send-receive",
        ui_args={
            "title": "Voice Agent",
            "subtitle": f"TTS: {TTS_BACKEND}" + (
                f" — select a Fish voice below, or '(default)' for the baked-in voice"
                if TTS_BACKEND == "fish" else ""
            ),
        },
    )

    # Add the "Add voice clone" accordion into the generated Blocks.
    # Re-entering the Blocks context lets us append more components after
    # FastRTC has finished building its default layout.
    if TTS_BACKEND == "fish" and voice_dd is not None:
        with stream.ui:
            with gr.Accordion("🎙️ Add a voice clone (10–30s of speech)", open=False):
                gr.Markdown(
                    "Record or upload a clip, fill in the transcript (or click "
                    "**Auto-transcribe**), give it a short ID, then **Save**. "
                    "The voice will appear in the dropdown above."
                )
                new_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Reference audio",
                )
                with gr.Row():
                    new_id = gr.Textbox(
                        label="Voice ID",
                        placeholder="e.g. sarah_sample1 (alphanumeric, -, _, space)",
                        scale=2,
                    )
                    transcribe_btn = gr.Button("Auto-transcribe", scale=1)
                new_transcript = gr.Textbox(
                    label="Transcript (exact words spoken)",
                    lines=3,
                    placeholder="What is said in the reference clip",
                )
                save_btn = gr.Button("Save voice clone", variant="primary")
                save_status = gr.Textbox(
                    label="Status", interactive=False, lines=1,
                )

                def _do_transcribe(audio_path):
                    if not audio_path:
                        return gr.update(), "Record or upload audio first."
                    t = whisper_transcribe_file(audio_path)
                    if not t:
                        return gr.update(), "Transcribe failed — check logs."
                    return t, f"Transcribed ({len(t)} chars)."

                def _do_save(audio_path, transcript, voice_id):
                    ok, msg = fish_save_voice(audio_path, transcript, voice_id)
                    if ok:
                        ids = fish_list_voices()
                        # Also activate the new voice right away
                        set_fish_voice(voice_id.strip())
                        return msg, gr.update(
                            choices=["(default)"] + ids,
                            value=voice_id.strip(),
                        )
                    return msg, gr.update()

                transcribe_btn.click(
                    _do_transcribe,
                    inputs=[new_audio],
                    outputs=[new_transcript, save_status],
                )
                save_btn.click(
                    _do_save,
                    inputs=[new_audio, new_transcript, new_id],
                    outputs=[save_status, voice_dd],
                )

    return stream


if __name__ == "__main__":
    prewarm()
    stream = build_stream()
    auth = os.environ.get("GRADIO_AUTH")
    auth_pairs = None
    if auth:
        auth_pairs = [tuple(pair.split(":")) for pair in auth.split(",")]
        logger.info(f"Auth enabled for {len(auth_pairs)} user(s)")

    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False,
        show_error=True,
        auth=auth_pairs,
    )
