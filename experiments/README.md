# Experiments

Loose ML experiment scripts on Blackwell GPUs — fast iteration, Gradio UIs, benchmarks.

## Active Experiments

| Experiment | Model | What it does | Run command |
|------------|-------|-------------|-------------|
| **[pixel-gen](pixel-gen/)** | SDXL + Lightning LoRA | Pixel art generation with optimized cleanup pipeline. Fused Lightning (4-step, ~0.25s), Tiny VAE, BOX downscale, early-stopping palette. Shipped to mythxengine as `pixelgen` microservice. | `CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/pixel-gen/app.py` |
| **[flux2](flux2/)** | FLUX.2 Klein 9B | Distilled image generation Gradio demo. 9B param model, ~29GB on disk. | `CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/flux2/demo.py` |
| **[ltx-video](ltx-video/)** | LTX-2.3 | Text-to-video generation — Gradio demo, benchmark script, and vLLM-Omni API variant. Fast/Balanced/Quality mode toggle. | `CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/ltx-video/gradio-demo.py` |
| **[qwen3-omni](qwen3-omni/)** | Qwen3-Omni | Multimodal (audio+vision+text) demo adapted for local vLLM-Omni endpoint. | `uv run python -u experiments/qwen3-omni/app.py` |
| **[tts-compare](tts-compare/)** | Voxtral 4B / Fish S2 Pro / Kokoro 82M | A/B/C TTS comparison — same text, side-by-side playback with latency + RTF. Kokoro in-process, others as services. | `uv run python -u experiments/tts-compare/app.py` |
| **[stt-whisper](stt-whisper/)** | Whisper large-v3-turbo / distil-large-v3 / large-v3 | Speech-to-text comparison — upload or mic record, transcribe with speed metrics. Models auto-swap. | `CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/stt-whisper/app.py` |

## Backlog

See [TODO.md](TODO.md) for models and capabilities queued for evaluation.

### Image Generation
- **Anima** (Cosmos 2B) — anime/illustration via ComfyUI
- **Z-Image** (6B) — foundation model, rich aesthetics
- **Z-Image-Turbo** (6B distilled) — 8-step fast photorealistic

### OCR / Document Understanding
- **Qianfan-OCR** (5B) — image-to-Markdown, tables, charts
- **GLM-OCR** (0.9B) — lightweight, MIT license

### Speech
- **Voxtral Mini 4B** — real-time STT, 13 languages

## Conventions

- Each experiment is a self-contained directory with its own scripts
- Gradio UIs bind to `0.0.0.0` on unique ports (7860+)
- Use `CUDA_VISIBLE_DEVICES=1` to run on GPU 1 (GPU 0 is vLLM)
- Outputs go to `/mnt/data/comfyui/output/{experiment}/`
- Models route through `HF_HOME=/mnt/models/huggingface`
- Dependencies shared via `experiments/pyproject.toml` (uv workspace member)
