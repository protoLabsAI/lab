# Voice Agent

Real-time conversational voice agent on Blackwell GPU.

```
Mic → [Silero VAD] → [Whisper Turbo STT] → [Qwen 4B LLM] → [Kokoro TTS] → Speaker
        ~1ms              ~80ms                ~160ms           ~50ms
```

## Benchmarks (RTX PRO 6000 Blackwell, Qwen3.5-4B-Int4 + Kokoro 82M)

| Stage | Avg | Low | High |
|-------|:---:|:---:|:----:|
| **STT** (Whisper large-v3-turbo) | 80ms | 60ms | 100ms |
| **LLM** (Qwen3.5-4B-Int4, 297 tok/s) | 160ms | 100ms | 250ms |
| **TTS** (Kokoro 82M) | 50ms | 40ms | 70ms |
| **Total end-to-end** | **290ms** | **240ms** | **390ms** |

Cold start (first turn, model loading): ~3.4s. All subsequent turns sub-400ms.

**290ms average is faster than human conversational turn-taking latency (~300ms).**

## Components

| Stage | Model | Params | VRAM |
|-------|-------|--------|:----:|
| VAD | Silero VAD | 1.8M | CPU |
| STT | whisper-large-v3-turbo | 809M | ~6GB |
| LLM | Qwen3.5-4B-Int4 via vLLM | 4B | ~4GB (GPU 0) |
| TTS | Kokoro 82M | 82M | ~2GB |

Total VRAM: ~8GB on GPU 1 (STT + TTS), ~4GB on GPU 0 (LLM via vLLM).

## Run

```bash
# Start LLM on GPU 0
bash models/vllm-swap.sh qwen-4b-int4

# Launch voice agent on GPU 1
CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/voice-agent/app.py

# For remote access with mic (HTTPS required for WebRTC):
sudo tailscale funnel 7866
# Then: https://protolabs.taild25506.ts.net/
```

UI at `http://localhost:7866` (local) or via Tailscale Funnel (remote).

## Auth

Set `GRADIO_AUTH` for login protection:
```bash
GRADIO_AUTH="user:pass,user2:pass2" uv run python -u experiments/voice-agent/app.py
```

## TTS Backends

Toggle in the UI:

- **Kokoro** (default): 82M params, ~50ms, in-process, Apache 2.0. Best latency.
- **Voxtral 4B**: ~250ms, external service on :8091, voice cloning, richer prosody.

## Architecture

Uses [FastRTC](https://github.com/gradio-app/fastrtc) with `ReplyOnPause` (Silero VAD) for WebRTC audio streaming. Pipeline is cascaded — each stage fires on completion of the previous. LLM thinking mode disabled for direct responses.

## What's next

- Stream LLM tokens to TTS (don't wait for full response) — could shave 50-100ms
- Voxtral streaming TTS (`/v1/audio/speech/stream`) for lower perceived latency
- Qwen3.5-9B-MTP (112 tok/s) for better reasoning quality at ~50ms extra latency
- Conversation memory / RAG for context-aware responses
