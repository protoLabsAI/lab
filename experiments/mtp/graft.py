#!/usr/bin/env python
"""Graft a Qwen3.5 MTP head from a donor checkpoint into a target fine-tune.

The MTP head is the 15 ``mtp.*`` tensors a native Qwen3.5 checkpoint ships but that
many fine-tunes (e.g. DeepReinforce's Ornith-1.0) drop. They are *self-contained*
(they don't reference the base model's tensor names; the coupling is purely at runtime
where ``fc`` fuses the base hidden state with the next-token embedding), so grafting is a
verbatim copy of those 15 tensors into the target's safetensors set.

This does NOT rewrite the target's big shards: it writes one new ``model-mtp.safetensors``
shard with the head and patches ``model.safetensors.index.json``. Everything else
(original shards, config, tokenizer, processor) is hard-linked/copied unchanged.

The grafted head is initialized from the donor (Qwen). It will *load and serve* (shapes
match) but acceptance will be poor until distilled against the target's own hidden states
(see distill.py) -- the head is co-trained with the residual stream, and a fine-tune moved
that stream. Measuring naive-graft acceptance is itself a useful baseline.

Usage:
  python graft.py --donor Qwen/Qwen3.5-9B \
                  --target deepreinforce-ai/Ornith-1.0-9B \
                  --out /mnt/data/checkpoints/ornith-9b-mtp-graft
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_lib import MTP_TENSORS, snapshot_dir, validate_graft_compat  # noqa: E402

MTP_SHARD = "model-mtp.safetensors"


def link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(os.path.realpath(src), dst)  # hardlink to save space/time
    except OSError:
        shutil.copy2(os.path.realpath(src), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--donor", required=True, help="HF repo id or path of a checkpoint that HAS the mtp.* head")
    ap.add_argument("--target", required=True, help="HF repo id or path of the fine-tune to graft INTO")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--dtype", default="keep", choices=["keep", "bfloat16", "float32"],
                    help="dtype for the grafted head (default: keep donor dtype; bf16 is recommended even on FP8 bases)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    donor = snapshot_dir(args.donor)
    target = snapshot_dir(args.target)
    print(f"donor : {args.donor}\n        {donor}")
    print(f"target: {args.target}\n        {target}")

    problems = validate_graft_compat(donor, target)
    if problems:
        print("\nINCOMPATIBLE:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"\ncompatible: donor ships all {len(MTP_TENSORS)} mtp.* tensors; target can host them.")

    if os.path.exists(args.out):
        if not args.force:
            print(f"refusing to overwrite existing {args.out} (use --force)")
            return 1
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    # 1. Copy every non-weight file (config, tokenizer, processor, generation_config, ...)
    #    and the original weight shards + index unchanged.
    target_index_path = os.path.join(target, "model.safetensors.index.json")
    if not os.path.exists(target_index_path):
        print("ERROR: target has no model.safetensors.index.json (single-file checkpoints not yet supported)")
        return 1
    index = json.load(open(target_index_path))
    weight_map: dict[str, str] = index["weight_map"]

    for f in os.listdir(target):
        src = os.path.join(target, f)
        if not os.path.isfile(os.path.realpath(src)):
            continue
        link_or_copy(src, os.path.join(args.out, f))

    # 2. Pull the 15 mtp.* tensors out of the donor.
    donor_shards = glob.glob(os.path.join(donor, "*.safetensors"))
    dtype = {"keep": None, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    head: dict[str, torch.Tensor] = {}
    for shard in donor_shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k in MTP_TENSORS:
                    t = f.get_tensor(k)
                    if dtype is not None:
                        t = t.to(dtype)
                    head[k] = t.contiguous()
    got = sorted(head)
    assert got == sorted(MTP_TENSORS), f"extracted {len(got)} != 15 mtp tensors: {got}"
    print(f"extracted {len(head)} mtp.* tensors from donor (dtype={args.dtype})")

    # 3. Write the new mtp shard + patch the index.
    save_file(head, os.path.join(args.out, MTP_SHARD), metadata={"format": "pt"})
    added_bytes = 0
    for k, t in head.items():
        weight_map[k] = MTP_SHARD
        added_bytes += t.numel() * t.element_size()
    index.setdefault("metadata", {})
    if "total_size" in index["metadata"]:
        index["metadata"]["total_size"] += added_bytes
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    print(f"\nwrote grafted checkpoint -> {args.out}")
    print(f"  + {MTP_SHARD} ({added_bytes/1e6:.1f} MB), index patched (+{len(head)} entries)")
    print("\nNEXT: distill the head against the target's own hidden states (distill.py),")
    print("      then serve with --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
