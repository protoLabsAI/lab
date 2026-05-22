# PARKED — voice-agent

Parked 2026-05-22. ORBIS retired; protoVoice handles the studio's voice surface now. See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

Real-time voice loop on Blackwell: Silero VAD → Whisper-Turbo STT → Qwen 4B INT4 LLM (streaming) → Kokoro TTS. Tailscale-served on `protolabs.taild25506.ts.net`.

## Where it stood

- v3 (prewarm) shipped: **165 ms TTFA, 210 ms end-to-end** on the Kokoro pipeline
- Fish S2 streaming variant integrated, voice cloning + in-UI record/transcribe/save working (see [project_voice_agent_fish.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_voice_agent_fish.md))
- Tailscale HTTPS cert + key checked into the dir (rotate before any reuse — `protolabs.taild25506.ts.net.{crt,key}`)

## Why parked

The brand surface for voice is `protoVoice` at `voice.proto-labs.ai`, owned by the homelab-iac/voice stack. This experiment was an ORBIS feeder; without ORBIS there is no consumer.

## How to resume

The 165 ms / 210 ms numbers and the streaming-LLM → sentence-chunker → chunked-TTS pattern are protolabs.studio breakdown material in themselves. If a new voice surface needs a substrate, the prewarm-everything-on-startup pattern in `app.py` is the load-bearing trick — port that, don't resurrect the whole gradio app.

Rotate the Tailscale cert before any reuse.
