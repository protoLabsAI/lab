# PARKED — ltx-video

Parked 2026-05-22. Video gen is not on the beach head. See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

LTX-2.3 22B video gen on Blackwell. Distilled (fp8, 8+4 steps, CFG 1.0) for fast drafts, dev (40 steps, CFG 3.0–3.5) for quality, two-stage with upsampler for best output. Gradio demo at `gradio-demo.py`, vLLM-omni variant at `gradio-demo-vllm-omni.py`. Benchmarks for 5–60 s clips on a single 96 GB card.

## How to resume

Video output isn't a substrate; it's stock material. If a studio experience (a game trailer, a breakdown b-roll) needs generated video, run the script — don't reopen this as an experiment. Models on `/mnt/models/huggingface/Lightricks/LTX-2.3/` (~97 GB) stay until disk pressure forces reclaim.

Blackwell constraints in `README.md` (no xformers, FA4 forward-only, torch.compile not integrated) are the durable finding.
