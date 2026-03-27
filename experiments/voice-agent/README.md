# Voice Agent

Real-time conversational voice agent on Blackwell GPU.

```
Mic → [Silero VAD] → [Whisper Turbo STT] → [Qwen 4B LLM streaming] → [TTS chunked] → Speaker
        ~1ms              ~55ms                 ~150ms                    ~95ms
```

## Benchmarks (RTX PRO 6000 Blackwell)

### Kokoro Pipeline (speed ceiling)

Qwen3.5-4B-Int4 (297 tok/s) + Kokoro 82M, streaming LLM→TTS with prewarm.

| Metric | Avg | Low | High |
|--------|:---:|:---:|:----:|
| **TTFA** (time to first audio) | **165ms** | 150ms | 180ms |
| STT (Whisper large-v3-turbo) | 55ms | 50ms | 60ms |
| LLM (Qwen3.5-4B, streaming) | 150ms | 140ms | 160ms |
| TTS (Kokoro 82M, chunked) | 95ms | 90ms | 100ms |
| **Total end-to-end** | **210ms** | 190ms | 230ms |
| First turn (pre-warmed) | 180ms | — | — |
| First turn (cold, no prewarm) | 3,200ms | — | — |

**165ms TTFA is faster than human conversational turn-taking (~300ms).**

### Optimization History

| Version | TTFA | Total | What changed |
|---------|:----:|:-----:|-------------|
| v1 (sequential) | 290ms | 290ms | Baseline — full LLM response then full TTS |
| v2 (streaming) | 170ms | 230ms | Stream LLM tokens → sentence chunker → chunked TTS |
| v3 (prewarm) | **165ms** | **210ms** | Pre-warm Whisper + Kokoro + LLM prefix cache on startup |

## Components

| Stage | Model | Params | VRAM |
|-------|-------|--------|:----:|
| VAD | Silero VAD | 1.8M | CPU |
| STT | whisper-large-v3-turbo | 809M | ~6GB |
| LLM | Qwen3.5-4B-Int4 via vLLM | 4B | ~4GB (GPU 0) |
| TTS (fast) | Kokoro 82M | 82M | ~2GB |
| TTS (quality) | Voxtral 4B | 4.1B | external (port 8091) |

## Run

```bash
# Start LLM on GPU 0
bash models/vllm-swap.sh qwen-4b-int4

# Launch voice agent on GPU 1 (pre-warms all models on startup ~3.3s)
CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/voice-agent/app.py

# For remote access with mic (HTTPS required for WebRTC):
sudo tailscale funnel 7866
# Then: https://protolabs.taild25506.ts.net/
```

## Auth

```bash
GRADIO_AUTH="user:pass,user2:pass2" uv run python -u experiments/voice-agent/app.py
```

## Streaming Pipeline Architecture

```
LLM tokens → SentenceChunker → TTS synthesis → Audio playback
  (stream)    (adaptive split)   (per chunk)    (immediate yield)
```

- **Adaptive chunking**: first chunk splits aggressively (comma-level, 10 char min) for fast TTFA, subsequent chunks wait for sentence boundaries (30 char min) for natural prosody
- **Overlapped**: audio plays while LLM still generates + next chunk synthesizes
- **Pre-warmed**: Whisper CUDA kernels, Kokoro vocoder, and LLM prefix cache all warmed on startup — eliminates 3s cold start

## TTS Backends

Toggle in the UI:

- **Kokoro** (default): 82M params, ~95ms/chunk, in-process, Apache 2.0
- **Voxtral 4B**: ~250ms/chunk, external service on :8091, voice cloning, Qwen text encoder

## What's Next

- Voxtral streaming TTS (`/v1/audio/speech/stream`) for quality upgrade
- Qwen3.5-9B-MTP (112 tok/s) for better reasoning at ~50ms extra
- Barge-in / interruption handling (cancel TTS on new speech)
- Conversation memory / RAG
