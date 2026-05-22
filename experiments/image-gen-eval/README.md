# Image gen eval

5-task A/B harness for evaluating multi-faceted local image models: clean T2I, targeted edit, style transfer, multi-turn identity preservation, compositional instruction.

## Tasks

1. **Clean T2I** — Polaroid of engraved brass diving helmet (text rendering + photoreal)
2. **Targeted edit** — red car → navy blue, everything else identical (object isolation)
3. **Style transfer** — portrait → 1920s Art Deco poster (style swap + identity)
4. **Multi-turn identity** — portrait → +glasses → +sweater → alpine scene (identity drift over 3 edits)
5. **Compositional** — watercolor park bench → add woman reading + dog + 'SUNDAY' sign

## Run

```bash
cd ~/dev/lab/experiments/image-gen-eval

# both models, all tasks
~/dev/lab/.venv/bin/python run_eval.py --model both

# just Qwen (can edit; handles all 5 tasks end to end)
~/dev/lab/.venv/bin/python run_eval.py --model qwen

# just FLUX.2-klein (gen-only here; base images only on edit tasks)
~/dev/lab/.venv/bin/python run_eval.py --model flux-klein

# subset
~/dev/lab/.venv/bin/python run_eval.py --model qwen --tasks 1,4
```

## Output

Writes to `/mnt/data/image-gen-eval/<YYYYMMDD-HHMMSS>/`:

- `qwen/task<N>_*.png` — Qwen-Image-Edit outputs (base + 1–3 edit turns)
- `flux-klein/task<N>_*.png` — FLUX.2-klein base generations (edit turns skipped)
- `meta.json` — prompts, seconds per turn, paths
- `grid.html` — side-by-side comparison view. Open in a browser:
  `http://protolabs:8188/...` won't serve it, so either `scp` or open locally / via a tiny `python -m http.server`.

## Scoring

Manual, ImgEdit-Bench style. For each task, rate 1–5:

| Axis | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Instruction adherence | ✓ | ✓ | ✓ | ✓ | ✓ |
| Edit quality (how clean) | — | ✓ | ✓ | ✓ | ✓ |
| Detail preservation (background, non-edited regions) | — | ✓ | — | ✓ | ✓ |
| Identity drift (0 = perfect, 5 = unrecognizable) | — | — | ✓ | ✓ × 3 | ✓ |
| Overall aesthetic | ✓ | ✓ | ✓ | ✓ | ✓ |

Record in `scores.yaml` next to `meta.json`.

## Caveats

- FLUX.2-klein (the warm variant on disk) is gen-only in this harness. For a like-for-like editor comparison, download FLUX.2 [dev] FP8 or FLUX.1 Kontext, wire up a second edit pipeline class. Today's harness accepts that FLUX won't do tasks 2–5 edit turns.
- Single-seed runs. For a real benchmark bump `--runs N` (not implemented yet) and average.
- Manual scoring. Could swap to GPT-4o judge or an ImageReward model for automated scoring in a later pass.
