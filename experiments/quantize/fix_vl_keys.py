#!/usr/bin/env python3
"""Repair mangled tensor keys in quantized VL checkpoints.

llm-compressor's save path on Qwen3_5ForConditionalGeneration (loaded via
AutoModelForImageTextToText) emits duplicated module prefixes:
  model.language_model.language_model.language_model.layers.*  (LM tensors)
  model.language_model.visual.*                                 (vision tensors)
transformers reloads its own mangled structure fine, but vLLM maps canonical
names, silently skips these tensors, and serves garbage from uninitialized
params. Canonical layout (matches the source checkpoints):
  model.language_model.layers.* / embed_tokens / norm
  model.visual.*
  lm_head.weight

Rewrites model.safetensors in place (atomic via temp file) and fixes the
quantization_config ignore list in config.json to match.

Usage: python fix_vl_keys.py <artifact_dir> [...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def canon(key: str) -> str:
    key = re.sub(r"^model\.(language_model\.)+visual\.", "model.visual.", key)
    key = re.sub(r"^model\.(language_model\.)+", "model.language_model.", key)
    return key


def fix_dir(d: Path):
    st = d / "model.safetensors"
    if not st.exists():
        print(f"{d}: no single-file model.safetensors (sharded not supported), skip")
        return

    tensors, renamed = {}, 0
    with safe_open(st, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            nk = canon(k)
            if nk != k:
                renamed += 1
            if nk in tensors:
                raise SystemExit(f"{d}: rename collision on {nk} — aborting, no changes made")
            tensors[nk] = f.get_tensor(k)

    cfg_path = d / "config.json"
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config") or cfg.get("compression_config") or {}
    ign = qc.get("ignore")
    ign_fixed = 0
    if isinstance(ign, list):
        new_ign = []
        for name in ign:
            nn = canon(name)
            ign_fixed += nn != name
            new_ign.append(nn)
        qc["ignore"] = new_ign

    if not renamed and not ign_fixed:
        print(f"{d.name}: already canonical")
        return

    tmp = st.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(st)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"{d.name}: renamed {renamed} tensors, fixed {ign_fixed} ignore entries")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        fix_dir(Path(arg))
