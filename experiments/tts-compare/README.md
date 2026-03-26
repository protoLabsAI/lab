# TTS Compare

A/B/C comparison of three text-to-speech models — same text, side-by-side playback with latency and RTF metrics.

| Model | Params | RTF (Blackwell) | VRAM | Voice Cloning | License |
|-------|--------|:---------------:|:----:|:-------------:|---------|
| **Voxtral 4B** | 4.1B | ~0.23 | ~35GB | Yes (3s min) | CC BY-NC 4.0 |
| **Fish Audio S2 Pro** | 4.4B | ~3.0 | ~22GB | Yes (10s min) | Non-commercial |
| **Kokoro 82M** | 82M | ~0.01 | ~2GB | No (54 fixed voices) | Apache 2.0 |

## Run

```bash
# 1. Start TTS backends (need GPU)

# Voxtral (GPU 0, ~35GB VRAM)
CUDA_VISIBLE_DEVICES=0 vllm-omni serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8091

# Fish Audio S2 Pro (GPU 1, ~22GB VRAM)
cd ~/dev/fish-speech && CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.api_server \
  --listen 0.0.0.0:8092 \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --decoder-config-name modded_dac_vq

# 2. Start Gradio UI (no GPU needed, Kokoro runs in-process)
cd ~/dev/lab
uv run python -u experiments/tts-compare/app.py
```

UI at `http://ava-ai:7864`.

Kokoro runs in-process (82M params, loads in ~2s). Voxtral and Fish Audio run as external services — both can't share a GPU so they need separate GPUs or sequential testing.

## Notes

- **Voxtral** needs `vllm-omni` (installed in `~/dev/vllm-env/`). Requires vllm 0.18+.
- **Fish Audio** uses its own venv at `~/dev/fish-speech/.venv/`.
- **Kokoro** installs via `pip install kokoro`. Needs `espeak-ng` system package and `en_core_web_sm` spacy model.
- Audio files save to `/mnt/data/comfyui/output/tts-compare/` with model name in the filename for easy sharing.
