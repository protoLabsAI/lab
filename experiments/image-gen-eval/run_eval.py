"""
Image generation + editing A/B eval harness.

Runs a 5-task mini-eval across any image model variant (base gen + editor).
Primary target: Qwen-Image-Edit-2511 (gen + edit, multi-turn identity preservation).
Challenger: FLUX.2-klein (gen only — editing tasks skipped on this path).

Tasks (see task definitions below):
  1. Clean text-to-image (quality + text rendering)
  2. Targeted edit — change one property, keep rest identical
  3. Style transfer while preserving identity
  4. Multi-turn identity edit chain (3 turns, measure drift)
  5. Compositional instruction (two inputs → composite scene)

Outputs: /mnt/data/image-gen-eval/<timestamp>/<model>/task<N>_<turn>.png
         /mnt/data/image-gen-eval/<timestamp>/grid.html  (side-by-side comparison)
         /mnt/data/image-gen-eval/<timestamp>/meta.json   (timings, seeds, prompts)

Usage:
  python run_eval.py --model qwen
  python run_eval.py --model flux-klein
  python run_eval.py --model both              # run both sequentially
  python run_eval.py --model both --tasks 1,2  # subset of tasks
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


# --------------------------- task definitions ---------------------------

@dataclass
class Task:
    num: int
    name: str
    kind: str                       # "t2i" | "edit" | "multi_turn" | "compose"
    prompt: str | None = None       # for t2i
    base_prompt: str | None = None  # used to generate the seed image for edit tasks
    edit_prompts: list[str] = field(default_factory=list)  # for edit / multi_turn / compose


TASKS: list[Task] = [
    Task(
        num=1, name="clean_t2i", kind="t2i",
        prompt="A vintage Polaroid photograph of a brass diving helmet resting on a wet wooden pier at golden hour. "
               "The engraved word 'NAUTILUS' is clearly visible on the faceplate. Sharp focus, film grain, "
               "warm sunset light reflecting off brass.",
    ),
    Task(
        num=2, name="targeted_edit", kind="edit",
        base_prompt="A clean studio photograph of a bright red 1967 Mustang convertible parked on a paved driveway, "
                    "three-quarter front view, daylight, crisp focus.",
        edit_prompts=[
            "Change only the color of the car to deep navy blue. Keep everything else — the background, lighting, "
            "shadows, position, and all other details — completely identical."
        ],
    ),
    Task(
        num=3, name="style_transfer", kind="edit",
        base_prompt="A waist-up studio photograph of a woman in her thirties with shoulder-length dark hair, "
                    "wearing a plain grey t-shirt, looking directly at the camera, neutral expression, soft studio lighting, "
                    "plain backdrop.",
        edit_prompts=[
            "Redraw this portrait in the style of a 1920s Art Deco travel poster. Preserve the person's identity — "
            "same face, same hair, same gender — but render with bold geometric shapes, gold and teal color palette, "
            "stylized flat shading, period-appropriate typography framing. Not a photograph anymore."
        ],
    ),
    Task(
        num=4, name="multi_turn_identity", kind="multi_turn",
        base_prompt="A waist-up photograph of a man in his forties with short brown hair and a trimmed beard, "
                    "wearing a plain black t-shirt, neutral expression, plain grey studio backdrop, soft even lighting.",
        edit_prompts=[
            "Add round wire-rimmed glasses to his face. Nothing else should change.",
            "Change his t-shirt to a chunky cream-colored wool sweater. Keep his face, glasses, hair, and "
            "expression identical.",
            "Place him in an alpine snowy scene at golden hour — distant snowy peaks, evergreens, soft warm "
            "backlight. Keep him, his face, glasses, sweater, and expression identical; only the background changes.",
        ],
    ),
    Task(
        num=5, name="compositional", kind="compose",
        base_prompt="A watercolor illustration of a wooden park bench in a grassy park at sunset, warm golden "
                    "light, no people or animals visible, soft brushstrokes, artistic mood.",
        edit_prompts=[
            "Add to this scene: a woman sitting on the park bench reading a book, and a golden retriever dog "
            "curled up asleep at her feet. Render everything in the same watercolor style, same lighting, same palette. "
            "Add a hand-lettered wooden sign in the foreground that reads 'SUNDAY'."
        ],
    ),
]


# --------------------------- pipeline helpers ---------------------------

def load_qwen_image_edit(device: str = "cuda:0") -> "QwenImageEditPipeline":
    from diffusers import QwenImageEditPipeline
    print("  loading Qwen-Image-Edit-2511 ...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_flux_klein(device: str = "cuda:0"):
    from diffusers import Flux2KleinPipeline
    print("  loading FLUX.2-klein-4B ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_qwen_edit(pipe, prompt: str, image: Image.Image | None = None, seed: int = 42) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    # QwenImageEditPipeline always requires `image`. For pure T2I we pass a
    # blank white canvas; the model treats it as "generate from scratch".
    if image is None:
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=30,
        generator=gen,
    )
    return result.images[0]


def run_flux_gen(pipe, prompt: str, seed: int = 42) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024, width=1024,
        generator=gen,
    )
    return result.images[0]


# --------------------------- eval orchestration ---------------------------

def run_model(model_name: str, tasks: list[Task], out_dir: Path, device: str = "cuda:0") -> dict:
    """Run the full task set against a single model. Returns per-task metadata."""
    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    # Load pipeline
    t_load0 = time.time()
    if model_name == "qwen":
        pipe = load_qwen_image_edit(device)
    elif model_name == "flux-klein":
        pipe = load_flux_klein(device)
    else:
        raise ValueError(f"unknown model: {model_name}")
    t_load = time.time() - t_load0
    print(f"  loaded in {t_load:.1f}s")

    try:
        for task in tasks:
            task_key = f"task{task.num}_{task.name}"
            print(f"\n[{model_name}] task {task.num}: {task.name} ({task.kind})")

            entry: dict = {"task": task.num, "name": task.name, "kind": task.kind, "turns": []}

            if task.kind == "t2i":
                t0 = time.time()
                img = (run_flux_gen if model_name == "flux-klein" else run_qwen_edit)(
                    pipe, task.prompt
                )
                dt = time.time() - t0
                path = model_out / f"{task_key}.png"
                img.save(path)
                entry["turns"].append({"prompt": task.prompt, "path": str(path), "seconds": dt})
                print(f"  done in {dt:.1f}s → {path.name}")

            elif task.kind in ("edit", "multi_turn", "compose"):
                # Step 0: generate the base image. Always use the same model for consistency.
                t0 = time.time()
                if model_name == "flux-klein":
                    base = run_flux_gen(pipe, task.base_prompt)
                else:
                    base = run_qwen_edit(pipe, task.base_prompt)
                dt0 = time.time() - t0
                base_path = model_out / f"{task_key}_base.png"
                base.save(base_path)
                entry["turns"].append({"prompt": task.base_prompt, "path": str(base_path), "seconds": dt0, "is_base": True})
                print(f"  base in {dt0:.1f}s → {base_path.name}")

                # FLUX.2-klein doesn't edit existing images in this pipeline — skip the edit turns.
                if model_name == "flux-klein":
                    entry["skipped_edits"] = "flux-klein is gen-only in this harness"
                    print(f"  [skip] flux-klein does not accept image input in this pipeline")
                else:
                    current = base
                    for i, prompt in enumerate(task.edit_prompts, start=1):
                        t1 = time.time()
                        edited = run_qwen_edit(pipe, prompt, image=current)
                        dt1 = time.time() - t1
                        edit_path = model_out / f"{task_key}_edit{i}.png"
                        edited.save(edit_path)
                        entry["turns"].append({"prompt": prompt, "path": str(edit_path), "seconds": dt1})
                        print(f"  edit {i} in {dt1:.1f}s → {edit_path.name}")
                        current = edited  # chain for multi-turn

            results[task_key] = entry

    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return {"model": model_name, "load_seconds": t_load, "tasks": results}


# --------------------------- HTML grid ---------------------------

def write_html_grid(out_dir: Path, meta: dict) -> Path:
    models = [m["model"] for m in meta["models"]]
    task_nums = sorted({int(k.split("_")[0][4:]) for m in meta["models"] for k in m["tasks"]})

    rows: list[str] = []
    for num in task_nums:
        task_keys_for_num = [
            k for m in meta["models"] for k in m["tasks"] if int(k.split("_")[0][4:]) == num
        ]
        tk = task_keys_for_num[0]
        task_name = meta["models"][0]["tasks"][tk]["name"] if tk in meta["models"][0]["tasks"] else tk
        kind = meta["models"][0]["tasks"][tk]["kind"] if tk in meta["models"][0]["tasks"] else "?"

        rows.append(f"<h2>Task {num}: {task_name} ({kind})</h2>")
        # table: one column per (model, turn)
        rows.append('<table border="1" cellpadding="4"><tr>')
        for m in meta["models"]:
            entry = m["tasks"].get(tk)
            if not entry:
                continue
            model_label = m["model"]
            for turn in entry["turns"]:
                rel = Path(turn["path"]).relative_to(out_dir)
                tname = "base" if turn.get("is_base") else "result"
                if "edit" in Path(turn["path"]).stem.split("_")[-1]:
                    tname = Path(turn["path"]).stem.split("_")[-1]
                rows.append(
                    f'<td valign="top" style="width:300px">'
                    f'<b>{model_label} — {tname}</b><br>'
                    f'<img src="{rel}" style="width:280px;height:auto"><br>'
                    f'<small>{turn["seconds"]:.1f}s</small><br>'
                    f'<details><summary>prompt</summary><pre style="white-space:pre-wrap">{turn["prompt"]}</pre></details>'
                    f'</td>'
                )
            if entry.get("skipped_edits"):
                rows.append(f'<td><i>skipped: {entry["skipped_edits"]}</i></td>')
        rows.append("</tr></table><hr>")

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Image gen eval</title>"
        "<style>body{font-family:system-ui;max-width:none;padding:20px;background:#fafafa}"
        "h1{margin-bottom:4px}h2{margin-top:24px}table{background:white}"
        "img{border:1px solid #ddd}pre{font-size:11px;max-width:280px}</style>"
        "</head><body>"
        f"<h1>Image gen eval — {meta['timestamp']}</h1>"
        f"<p>Models: {', '.join(models)}</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    path = out_dir / "grid.html"
    path.write_text(html)
    return path


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "flux-klein", "both"], default="both")
    ap.add_argument("--tasks", default="1,2,3,4,5", help="comma-separated task numbers")
    ap.add_argument("--out-root", default="/mnt/data/image-gen-eval")
    ap.add_argument("--out-dir", default=None, help="reuse a specific output dir (e.g. merge a rerun)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        stamp = out_dir.name
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    task_nums = {int(x) for x in args.tasks.split(",")}
    selected_tasks = [t for t in TASKS if t.num in task_nums]
    print(f"running {len(selected_tasks)} task(s): {[t.num for t in selected_tasks]}")
    print(f"output: {out_dir}")

    models_to_run = ["qwen", "flux-klein"] if args.model == "both" else [args.model]
    model_results: list[dict] = []

    for m in models_to_run:
        print(f"\n=== model: {m} ===")
        result = run_model(m, selected_tasks, out_dir, device=args.device)
        model_results.append(result)

    meta = {
        "timestamp": stamp,
        "device": args.device,
        "tasks": [t.num for t in selected_tasks],
        "models": model_results,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    grid = write_html_grid(out_dir, meta)
    print(f"\n✓ meta:  {out_dir / 'meta.json'}")
    print(f"✓ grid:  {grid}")
    print(f"\nview: file://{grid}")


if __name__ == "__main__":
    main()
