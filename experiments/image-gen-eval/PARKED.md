# PARKED — image-gen-eval

Parked 2026-05-22. Image-gen work moved to ava as `avaLab` substrate sibling; protoBanana is the consumer surface. See [project_protobanana_handoff.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_protobanana_handoff.md) and [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

5-task A/B harness comparing Qwen-Image-Edit-2511 vs FLUX.2-klein across clean T2I, targeted edit, style transfer, multi-turn identity, compositional. `run_eval.py` writes outputs + `grid.html` to `/mnt/data/image-gen-eval/<timestamp>/`.

## How to resume

Rebuild over there. The 5-task taxonomy is reusable; the runner is single-script and trivial to port. Outputs under `/mnt/data/image-gen-eval/` are local-only — if useful, copy to ava before reclaiming.
