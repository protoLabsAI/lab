# TTS Compare

A/B/C comparison of three text-to-speech models — same text, side-by-side playback with latency and RTF metrics. Includes a voice-cloning tab for Fish Audio S2 Pro.

| Model | Params | RTF (Blackwell) | VRAM | Voice Cloning | License |
|-------|--------|:---------------:|:----:|:-------------:|---------|
| **Voxtral 4B** | 4.1B | ~0.23 | ~35GB | Yes (3s min) | CC BY-NC 4.0 |
| **Fish Audio S2 Pro** | 4.4B | **0.40** | ~22GB | Yes (10s min) | Non-commercial |
| **Kokoro 82M** | 82M | ~0.01 | ~2GB | No (54 fixed voices) | Apache 2.0 |

Fish RTF figure is steady-state on ~15s of generated audio with `--half --compile` (see below). Initial warmup is ~2 min for torch.compile to codegen.

## Run

```bash
# 1. Start TTS backends (need GPU)

# Voxtral (GPU 0, ~35GB VRAM)
CUDA_VISIBLE_DEVICES=0 vllm-omni serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8091

# Fish Audio S2 Pro (GPU 1, ~22GB VRAM)
# NOTE: --half (bf16) + --compile are REQUIRED for acceptable RTF on Blackwell.
# Without them RTF is ~3.0; with them it's ~0.4. First call triggers ~2min compile warmup.
cd ~/dev/fish-speech && CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.api_server \
  --listen 0.0.0.0:8092 \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --decoder-config-name modded_dac_vq \
  --half \
  --compile

# 2. Start Gradio UI (no GPU needed, Kokoro runs in-process)
cd ~/dev/lab
uv run python -u experiments/tts-compare/app.py
```

UI at `http://ava-ai:7864`. Tabs: **Compare All**, **Voice Clone (Fish)**, **Single**.

Kokoro runs in-process (82M params, loads in ~2s). Voxtral and Fish Audio run as external services — both can't share a GPU so they need separate GPUs or sequential testing.

## Voice cloning (Fish S2 Pro)

The **Voice Clone (Fish)** tab supports three reference modes:

- **Inline (upload)** — upload or record 10–30s of reference audio + type the transcript. One-shot.
- **Saved reference** — pick from voices saved on the Fish server (`/v1/references/`).
- **Save new reference** — upload audio + transcript, assign an ID, persist on server.

Inline `[tag]` prompts are supported: `[pause]`, `[laughing]`, `[whisper]`, `[excited]`, `[professional broadcast tone]`, `[pitch up]`, etc. — 15,000+ free-form tags per the Fish S2 paper.

## Blackwell RTF notes

Out-of-the-box fish-speech serving on Blackwell RTX PRO 6000 sits at RTF ~3.0 (vs H200 claim of 0.195). With `--half --compile` we land at **0.40** (~7.6× speedup). Remaining gap to H200 is mostly memory bandwidth (HBM3e 4.8TB/s vs GDDR7 ~1.8TB/s) plus SGLang prefix caching which isn't in the open repo.

The community weight-only FP8 quant (`drbaph/s2-pro-fp8`) was tested: dequantizes to BF16 on forward pass, gives no meaningful speedup on Blackwell since FP8 tensor cores are never engaged. A real W8A8 via torchao `_scaled_mm` would be needed for actual FP8 compute — deferred for now.

## Notes

- **Voxtral** needs `vllm-omni` (installed in `~/dev/vllm-env/`). Requires vllm 0.18+.
- **Fish Audio** uses its own venv at `~/dev/fish-speech/.venv/`.
- **Kokoro** installs via `pip install kokoro`. Needs `espeak-ng` system package and `en_core_web_sm` spacy model.
- Audio files save to `/mnt/data/comfyui/output/tts-compare/` with model name in the filename for easy sharing.
