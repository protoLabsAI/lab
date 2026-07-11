"""Build the protoLabs agentic-distill corpus from sources.yaml.

  python build.py --version v0.1 --out /mnt/data/datasets/agentic-distill [--mix ornith|public|blend]

Pipeline: for each enabled source -> stream rows -> adapter -> Trajectory -> validate
-> dedup (prompt_hash) -> contamination-filter vs held-out eval -> weighted sample
-> write versioned JSONL (train.jsonl) + manifest.json (exact revisions + counts).

Reusable: the output is one JSONL of canonical Trajectory rows, portable to any student
and any trainer (LLaMA-Factory / TRL). Storage primary = /mnt/data/datasets (per CLAUDE.md);
optional private HF mirror via `hf upload --repo-type dataset --private` after.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# HF_HOME points at /mnt/models (12G free, 99% full). Route the raw dataset cache to
# /mnt/scratch (disposable, 500G+) so a big streaming pull can't fill the models drive.
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/scratch/hf-datasets-cache")

import yaml

import adapters
from schema import Trajectory

random.seed(0)  # reproducible sampling


def _contamination_hashes(held_cfg: dict) -> set[str]:
    """Prompt hashes of held-out eval prompts to exclude from train. TODO: load the
    actual τ²-bench prompts and hash them the same way Trajectory.prompt_hash does."""
    return set()  # wire once tau2-bench prompts are pulled


def build(version: str, out_root: str, mix: str = "blend", cap_per_source: int | None = None,
          only: set[str] | None = None):
    cfg = yaml.safe_load(Path(__file__).with_name("sources.yaml").read_text())
    out_dir = Path(out_root) / version
    out_dir.mkdir(parents=True, exist_ok=True)

    banned = _contamination_hashes(cfg.get("held_out_eval", {}))
    seen: set[str] = set()
    manifest: dict = {"version": version, "mix": mix, "sources": {}, "total": 0}
    n_total = 0

    with (out_dir / "train.jsonl").open("w") as fout:
        for key, scfg in cfg["sources"].items():
            if only and key not in only:
                continue
            if not only and not scfg.get("enabled"):
                continue
            if mix == "ornith" and scfg.get("teacher") != "ornith-35b":
                continue
            if mix == "public" and scfg.get("teacher") == "ornith-35b":
                continue
            scfg = {**scfg, "_key": key}
            adapter = adapters.get(key)
            kept = 0
            for row in _stream(scfg, cap_per_source):
                try:
                    for traj in adapter(row, scfg):
                        if traj.prompt_hash() in banned:      # contamination vs eval
                            continue
                        h = traj.content_hash()               # dedup on full content
                        if h in seen:
                            continue
                        seen.add(h)
                        fout.write(traj.model_dump_json() + "\n")
                        kept += 1
                        n_total += 1
                except NotImplementedError as e:
                    print(f"[skip] {key}: {e}")
                    break
                except Exception as e:
                    print(f"[warn] {key}: dropped a row — {e}")
            manifest["sources"][key] = {"kept": kept, "hf_id": scfg.get("hf_id"),
                                        "teacher": scfg.get("teacher"),
                                        "weight": scfg.get("weight", 1.0)}
            print(f"[ok] {key}: {kept} trajectories")

    manifest["total"] = n_total
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {n_total} trajectories -> {out_dir}/train.jsonl")


def _stream(scfg: dict, cap: int | None):
    """Yield raw rows for a source. Local Ornith JSONL vs HF download."""
    hf_id = scfg.get("hf_id")
    if not hf_id:  # locally-generated (ornith_tau)
        p = Path("/mnt/data/datasets/agentic-distill/_raw") / f"{scfg['_key']}.jsonl"
        if not p.exists():
            print(f"[skip] {scfg['_key']}: {p} not generated yet")
            return
        for i, line in enumerate(p.open()):
            if cap and i >= cap:
                break
            yield json.loads(line)
        return
    from datasets import get_dataset_split_names, load_dataset  # lazy: only when downloading
    # Not every set has a "train" split (e.g. orca is sharded by topic; smoltalk2 exposes
    # subsets as splits under a config). Manifest can pin `config:` and `split:`.
    cfg = scfg.get("config")
    split = scfg.get("split")
    try:
        if not split:
            avail = get_dataset_split_names(hf_id, cfg) if cfg else get_dataset_split_names(hf_id)
            split = "train" if "train" in avail else avail
        splits = [split] if isinstance(split, str) else split
        n = 0
        for sp in splits:
            for row in (load_dataset(hf_id, cfg, split=sp, streaming=True) if cfg
                        else load_dataset(hf_id, split=sp, streaming=True)):
                if cap and n >= cap:
                    return
                n += 1
                yield row
    except Exception as e:
        print(f"[skip] {scfg['_key']}: stream failed ({e}) — check hf_id/split/config")
        return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--out", default="/mnt/data/datasets/agentic-distill")
    ap.add_argument("--mix", choices=["ornith", "public", "blend"], default="blend")
    ap.add_argument("--cap-per-source", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated source keys to build (ignores enabled/mix)")
    a = ap.parse_args()
    build(a.version, a.out, a.mix, a.cap_per_source,
          only=set(a.only.split(",")) if a.only else None)
    # pyarrow/datasets streaming can SIGABRT in C++ destructors on teardown (after all
    # data is written). Hard-exit past it so a clean build doesn't report failure.
    import sys
    sys.stdout.flush()
    os._exit(0)
