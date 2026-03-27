# Voice Agent

Real-time conversational voice agent on Blackwell GPU.

```
Mic → [Silero VAD] → [Whisper Turbo STT] → [Qwen LLM] → [TTS] → Speaker
        ~1ms              ~120ms              ~90ms       ~50-250ms
```

Estimated end-to-end latency: **~700-900ms** (sub-second conversation).

## Components

| Stage | Model | Latency | VRAM |
|-------|-------|:-------:|:----:|
| VAD | Silero VAD | ~1ms (CPU) | 0 |
| STT | whisper-large-v3-turbo | ~120ms | ~6GB |
| LLM | Qwen3.5 via vLLM | ~90ms TTFT | shared (port 8000) |
| TTS (fast) | Kokoro 82M | ~50ms | ~2GB |
| TTS (quality) | Voxtral 4B | ~250ms | external (port 8091) |

## Run

```bash
# Ensure vLLM is running (any Qwen3.5 model)
bash models/vllm-swap.sh qwen-4b-int4   # fastest for voice agent

# Launch voice agent on GPU 1
CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/voice-agent/app.py

# Optional: start Voxtral for higher quality TTS
CUDA_VISIBLE_DEVICES=0 vllm-omni serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8091
```

UI at `http://protolabs:7866`. Speak into mic, agent responds when you pause.

## TTS Backends

Toggle in the UI:

- **Kokoro** (default): 82M params, ~50ms, in-process, Apache 2.0. Best for lowest latency.
- **Voxtral**: 4B params, ~250ms, external service, voice cloning, much richer prosody. Uses Qwen text encoder internally.

## Architecture

Uses [FastRTC](https://github.com/gradio-app/fastrtc) for WebRTC audio streaming with `ReplyOnPause` (Silero VAD). Pipeline is cascaded — each stage fires as soon as the previous one completes. LLM response is currently non-streaming (full response then TTS). Next step: stream LLM tokens to TTS for lower perceived latency.

## Latency Budget

| Component | Time |
|-----------|------|
| VAD silence detection | ~500ms (configurable) |
| Whisper STT | ~120ms |
| Qwen 4B TTFT + first clause | ~90ms |
| Kokoro TTS | ~50ms |
| **Total with Kokoro** | **~760ms** |
| **Total with Voxtral** | **~960ms** |
