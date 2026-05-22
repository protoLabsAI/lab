"""Scan an image-gen-eval output dir and rebuild meta.json + grid.html from files on disk.

Useful when a second run overwrote meta.json but the image files are still intact.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from run_eval import TASKS, write_html_grid


def scan_model_dir(model_dir: Path, task_defs: list) -> dict:
    tasks: dict[str, dict] = {}
    for task in task_defs:
        key = f"task{task.num}_{task.name}"
        entry: dict = {"task": task.num, "name": task.name, "kind": task.kind, "turns": []}

        if task.kind == "t2i":
            p = model_dir / f"{key}.png"
            if p.exists():
                entry["turns"].append({"prompt": task.prompt, "path": str(p), "seconds": 0.0})
        else:
            base = model_dir / f"{key}_base.png"
            if base.exists():
                entry["turns"].append({"prompt": task.base_prompt, "path": str(base), "seconds": 0.0, "is_base": True})
            for i, p in enumerate(task.edit_prompts, start=1):
                edit_path = model_dir / f"{key}_edit{i}.png"
                if edit_path.exists():
                    entry["turns"].append({"prompt": p, "path": str(edit_path), "seconds": 0.0})

        # only record the task if we found any image for it
        if entry["turns"]:
            tasks[key] = entry

    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    assert out_dir.exists(), out_dir

    models_found = []
    for sub in out_dir.iterdir():
        if not sub.is_dir():
            continue
        tasks = scan_model_dir(sub, TASKS)
        if tasks:
            models_found.append({"model": sub.name, "load_seconds": 0.0, "tasks": tasks})
            print(f"  {sub.name}: {len(tasks)} task(s), {sum(len(t['turns']) for t in tasks.values())} image(s)")

    meta = {
        "timestamp": out_dir.name,
        "device": "unknown (reconstructed)",
        "tasks": sorted({t["task"] for m in models_found for t in m["tasks"].values()}),
        "models": models_found,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    grid = write_html_grid(out_dir, meta)
    print(f"\n✓ meta:  {out_dir / 'meta.json'}")
    print(f"✓ grid:  {grid}")


if __name__ == "__main__":
    main()
