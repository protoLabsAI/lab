#!/usr/bin/env python3
"""Extract MTP draft-head tensors from a checkpoint into a bf16 sidecar.

Qwen3.5/3.6 checkpoints (and huihui abliterations of them) carry native
`mtp.*` tensors. Our quantization path loads the text model, which drops
them — and they should stay bf16 anyway. This pulls them into a
`model-mtp.safetensors` sidecar (same layout as protoLabsAI/Ornith-1.0-9B-MTP)
that vLLM loads via --speculative-config alongside the quantized base.

Usage:
  python extract_mtp.py --model huihui-ai/Huihui-Qwen3.5-9B-abliterated \
      --output /mnt/models/quantized/Huihui-Qwen3.5-9B-abliterated-NVFP4/model-mtp.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo id or local path")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(args.model)
    if not src.exists():
        src = Path(
            snapshot_download(args.model, allow_patterns=["*.safetensors*", "*.json"])
        )

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.load(open(index_path))["weight_map"]
        mtp_files = sorted({v for k, v in weight_map.items() if k.startswith("mtp.")})
        mtp_keys = [k for k in weight_map if k.startswith("mtp.")]
    else:
        mtp_files = ["model.safetensors"]
        mtp_keys = None  # discover below

    tensors = {}
    for fname in mtp_files:
        with safe_open(src / fname, framework="pt") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    tensors[k] = f.get_tensor(k)

    if not tensors:
        raise SystemExit(f"No mtp.* tensors found in {src}")
    if mtp_keys is not None and len(tensors) != len(mtp_keys):
        raise SystemExit(
            f"Expected {len(mtp_keys)} mtp tensors per index, extracted {len(tensors)}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    size_mb = out.stat().st_size / 1024**2
    print(f"{len(tensors)} mtp tensors → {out} ({size_mb:.0f} MB)")
    for k in sorted(tensors):
        print(f"  {k}  {tuple(tensors[k].shape)}  {tensors[k].dtype}")


if __name__ == "__main__":
    main()
