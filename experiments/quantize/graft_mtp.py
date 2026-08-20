#!/usr/bin/env python3
"""Graft the ThinkingCap MTP head into the heretic-NVFP4 artifact.

The abliterated base lost its MTP head, so after `thinkingcap_27b_heretic_requant.py`
quantizes the main weights, copy the already-extracted `model_mtp.safetensors` from the
base NVFP4 and merge its 15 `mtp.*` entries into the heretic index. MTP drafts / target
verifies → lossless even though the head came from the non-abliterated base.

  python graft_mtp.py \
    --src /mnt/models/quantized/ThinkingCap-Qwen3.6-27B-NVFP4 \
    --dst /mnt/models/quantized/ThinkingCap-Qwen3.6-27B-heretic-NVFP4
"""
import argparse, json, shutil, struct
from pathlib import Path

def st_header(path):
    """Parse a safetensors header → dict of tensor->meta (no __metadata__)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}

def st_bytes(path):
    return sum(m["data_offsets"][1] - m["data_offsets"][0] for m in st_header(path).values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base NVFP4 with model_mtp.safetensors")
    ap.add_argument("--dst", required=True, help="heretic NVFP4 to graft into")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)

    mtp_file = src / "model_mtp.safetensors"
    assert mtp_file.exists(), f"no MTP head at {mtp_file}"
    shutil.copy2(mtp_file, dst / "model_mtp.safetensors")

    idx_path = dst / "model.safetensors.index.json"
    if idx_path.exists():
        didx = json.load(open(idx_path))                    # sharded: merge into existing index
        wm = didx["weight_map"]
        total = didx.get("metadata", {}).get("total_size", 0)
    else:
        # single-file dst (26G fit one shard, no index) → synthesize one so vLLM loads BOTH files
        main = dst / "model.safetensors"
        assert main.exists(), f"no {main}"
        wm = {k: "model.safetensors" for k in st_header(main)}
        total = st_bytes(main)
        didx = {"metadata": {}, "weight_map": wm}
        print(f"synthesized index for single-file dst: {len(wm)} main tensors")

    mtp_map = {k: "model_mtp.safetensors" for k in st_header(mtp_file)}
    assert mtp_map, "no mtp.* tensors in head"
    wm.update(mtp_map)
    didx["weight_map"] = wm
    didx["metadata"]["total_size"] = total + st_bytes(mtp_file)
    json.dump(didx, open(idx_path, "w"), indent=2)
    print(f"grafted {len(mtp_map)} mtp tensors → {dst}/model_mtp.safetensors")
    print(f"index total_size = {didx['metadata']['total_size']:,}")

if __name__ == "__main__":
    main()
