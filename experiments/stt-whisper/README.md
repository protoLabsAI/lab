# STT Whisper

Speech-to-text comparison of Whisper model variants on Blackwell GPU.

| Model | Params | VRAM | Speed (est.) | Quality |
|-------|--------|:----:|:------------:|---------|
| **large-v3-turbo** | 809M | ~6GB | ~100x RT | Best speed/quality ratio |
| **distil-large-v3** | 756M | ~5GB | ~90x RT | Slightly lower quality |
| **large-v3** | 1.55B | ~10GB | ~50x RT | Best accuracy |

## Run

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -u experiments/stt-whisper/app.py
```

UI at `http://protolabs:7865`. Upload audio or record from mic. Models lazy-load on first use (~5s), then swap automatically when you change selection.

## Notes

- Uses PyTorch SDPA attention (no Flash Attention 2 needed, works on Blackwell)
- Batch size 24 default — with 96GB VRAM you can go higher
- Language auto-detection or manual selection
- Word-level timestamps optional
- Transcripts save to `/mnt/data/comfyui/output/stt-whisper/`
