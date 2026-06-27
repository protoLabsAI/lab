#!/usr/bin/env python3
"""
Low-memory native FP8 quantization — matches official Qwen FP8 format exactly.

Key format details (from Qwen/Qwen3.6-35B-A3B-FP8):
  - Block-wise FP8 E4M3 with [128, 128] scales
  - Scale tensors named `weight_scale_inv` (inverted: 1/scale), dtype bfloat16
  - MoE packed experts [N, rows, cols] unpacked to per-expert 2D tensors
  - Fused gate_up_proj [N, fused, hidden] split into gate_proj + up_proj

Processes one tensor at a time to minimize memory. Peak RAM ≈ 2× largest tensor.

Usage:
  python quantize_native_fp8_lowmem.py llmfan46/Qwen3.6-35B-A3B-uncensored-heretic
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/mnt/models/quantized")
BLOCK_SIZE = 128

SKIP_SUFFIXES = [
    "mlp.gate.weight",
    "shared_expert_gate.weight",
]

SKIP_PATTERNS = [
    "lm_head", "embed_tokens",
    "conv1d",
    "visual", "mtp",
    "layernorm", "layer_norm", "norm",
    ".A_log", ".D", ".dt_bias",
    # Keep the ENTIRE linear-attention / SSM path in bf16. The specific-name list
    # (in_proj_a/b) historically MISSED in_proj_qkv, in_proj_z, and out_proj — 2D
    # projections that would otherwise be FP8-quantized and corrupt the SSM
    # (Ornith-35B / Qwen3.5-MoE hybrid). Upstream FP8 ignores all `linear_attn.*`.
    "linear_attn",
    "in_proj_a", "in_proj_b",  # (redundant with linear_attn; kept for non-prefixed models)
]


def should_skip(name: str) -> bool:
    name_lower = name.lower()
    for s in SKIP_SUFFIXES:
        if name.endswith(s):
            return True
    for p in SKIP_PATTERNS:
        if p in name_lower:
            return True
    return False


def quantize_weight_blockwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to FP8 E4M3 with per-block [128,128] scaling.
    Returns (quantized_weight, scale_inv) in Qwen's weight_scale_inv convention.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    rows, cols = weight.shape

    pad_r = (BLOCK_SIZE - rows % BLOCK_SIZE) % BLOCK_SIZE
    pad_c = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad_r or pad_c:
        weight = torch.nn.functional.pad(weight.float(), (0, pad_c, 0, pad_r))

    r, c = weight.shape
    blocks = weight.reshape(r // BLOCK_SIZE, BLOCK_SIZE, c // BLOCK_SIZE, BLOCK_SIZE)
    blocks = blocks.permute(0, 2, 1, 3)

    block_max = blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    scale = block_max / fp8_max
    quantized = (blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    quantized = quantized.permute(0, 2, 1, 3).reshape(r, c)[:rows, :cols]
    # "weight_scale_inv" is a DeepSeek naming convention — vLLM expects the
    # DIRECT scale value (scale = max/fp8_max), NOT the mathematical inverse.
    scale_direct = scale.squeeze(-1).squeeze(-1).to(torch.bfloat16)

    return quantized, scale_direct


def process_tensor(name: str, tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor], int, int, list[str]]:
    """Process a single tensor. Returns (output_tensors, quantized, skipped, skipped_modules)."""
    output = {}
    quantized = 0
    skipped = 0
    skipped_modules: list[str] = []

    if should_skip(name) or tensor.ndim < 2:
        output[name] = tensor
        skipped += 1
        # Record the module name (strip .weight suffix) for modules_to_not_convert
        if tensor.ndim >= 2:
            module = name.rsplit(".", 1)[0] if "." in name else name
            skipped_modules.append(module)

    elif tensor.ndim == 3 and "experts." in name and "gate_up_proj" in name:
        num_experts = tensor.shape[0]
        half = tensor.shape[1] // 2
        for i in range(num_experts):
            gate = tensor[i, :half, :].float()
            q_gate, s_gate = quantize_weight_blockwise(gate)
            output[name.replace("experts.gate_up_proj", f"experts.{i}.gate_proj.weight")] = q_gate
            output[name.replace("experts.gate_up_proj", f"experts.{i}.gate_proj.weight_scale_inv")] = s_gate
            del gate, q_gate, s_gate

            up = tensor[i, half:, :].float()
            q_up, s_up = quantize_weight_blockwise(up)
            output[name.replace("experts.gate_up_proj", f"experts.{i}.up_proj.weight")] = q_up
            output[name.replace("experts.gate_up_proj", f"experts.{i}.up_proj.weight_scale_inv")] = s_up
            del up, q_up, s_up
            quantized += 2

    elif tensor.ndim == 3 and "experts." in name:
        num_experts = tensor.shape[0]
        proj_name = name.split("experts.")[-1]
        for i in range(num_experts):
            w = tensor[i].float()
            q_w, s_inv = quantize_weight_blockwise(w)
            output[name.replace(f"experts.{proj_name}", f"experts.{i}.{proj_name}.weight")] = q_w
            output[name.replace(f"experts.{proj_name}", f"experts.{i}.{proj_name}.weight_scale_inv")] = s_inv
            del w, q_w, s_inv
            quantized += 1

    elif tensor.ndim == 2:
        q_w, s_inv = quantize_weight_blockwise(tensor.float())
        output[name] = q_w
        if name.endswith(".weight"):
            output[name.replace(".weight", ".weight_scale_inv")] = s_inv
        else:
            output[name + "_scale_inv"] = s_inv
        del q_w, s_inv
        quantized += 1

    else:
        output[name] = tensor
        skipped += 1

    return output, quantized, skipped, skipped_modules


# Max bytes of output tensors per sub-shard before flushing to disk (~4GB)
SHARD_MAX_BYTES = 4 * 1024**3


def process_shard_streaming(
    shard_path: Path, output_dir: Path, shard_counter: list[int]
) -> tuple[dict[str, str], int, int, list[str]]:
    """Process a shard tensor-by-tensor. Writes multiple sub-shards to stay under memory.

    Returns (weight_map, quantized_count, skipped_count, skipped_modules).
    """
    logger.info(f"Processing {shard_path.name}...")

    weight_map = {}
    total_quantized = 0
    total_skipped = 0
    all_skipped_modules: list[str] = []

    pending_tensors = {}
    pending_bytes = 0

    def flush(is_final=False):
        nonlocal pending_tensors, pending_bytes
        if not pending_tensors:
            return
        shard_counter[0] += 1
        # Use placeholder name, will rename in main()
        out_name = f"model-{shard_counter[0]:05d}-of-PLACEHOLDER.safetensors"
        out_path = output_dir / out_name
        logger.info(f"  Writing sub-shard {out_name} ({len(pending_tensors)} tensors, {pending_bytes/1024**3:.1f}GB)")
        save_file(pending_tensors, str(out_path))
        for tname in pending_tensors:
            weight_map[tname] = out_name
        pending_tensors = {}
        pending_bytes = 0
        gc.collect()

    with safe_open(str(shard_path), framework="pt") as f:
        tensor_names = sorted(f.keys())
        total = len(tensor_names)

        for idx, name in enumerate(tensor_names):
            tensor = f.get_tensor(name)
            output, q_count, s_count, s_modules = process_tensor(name, tensor)
            del tensor

            total_quantized += q_count
            total_skipped += s_count
            all_skipped_modules.extend(s_modules)

            for tname, tval in output.items():
                pending_tensors[tname] = tval
                pending_bytes += tval.nelement() * tval.element_size()

            del output

            if pending_bytes >= SHARD_MAX_BYTES:
                flush()

            if (idx + 1) % 50 == 0:
                logger.info(f"  [{idx+1}/{total}] processed, pending {pending_bytes/1024**3:.1f}GB")

    flush(is_final=True)
    return weight_map, total_quantized, total_skipped, all_skipped_modules


def main():
    if len(sys.argv) < 2:
        print("Usage: python quantize_native_fp8_lowmem.py <model_id> [output_dir]")
        sys.exit(1)

    model_id = sys.argv[1]
    model_short = model_id.split("/")[-1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_ROOT / f"{model_short}-FP8"
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_home = Path(os.environ["HF_HOME"])
    model_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot = model_dir / "snapshots" / os.listdir(model_dir / "snapshots")[0]

    index_file = snapshot / "model.safetensors.index.json"
    if not index_file.exists():
        logger.error("No shard index found — single-file models not supported yet")
        sys.exit(1)

    index = json.load(open(index_file))
    shard_names = sorted(set(index["weight_map"].values()))
    logger.info(f"Found {len(shard_names)} shards with {len(index['weight_map'])} tensors")

    t0 = time.time()
    total_quantized = 0
    total_skipped = 0
    new_weight_map = {}
    shard_counter = [0]  # mutable counter for sub-shards
    all_skipped_modules: list[str] = []

    for shard_name in shard_names:
        shard_path = snapshot / shard_name
        wmap, q_count, s_count, s_modules = process_shard_streaming(shard_path, output_dir, shard_counter)
        total_quantized += q_count
        total_skipped += s_count
        new_weight_map.update(wmap)
        all_skipped_modules.extend(s_modules)

    quant_time = time.time() - t0
    num_shards = shard_counter[0]

    # Rename PLACEHOLDER shards to final names
    for i in range(1, num_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        (output_dir / old_name).rename(output_dir / new_name)
        # Update weight map
        for k, v in new_weight_map.items():
            if v == old_name:
                new_weight_map[k] = new_name

    all_shard_names = [f"model-{i:05d}-of-{num_shards:05d}.safetensors" for i in range(1, num_shards + 1)]

    # Write index
    total_size = sum((output_dir / s).stat().st_size for s in all_shard_names)
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, f, indent=2)

    # Build config.json from original + quantization_config
    orig_config = json.load(open(snapshot / "config.json"))
    # Deduplicate and sort modules_to_not_convert — tells vLLM which layers
    # to keep in bf16 (critical for Mamba/SSM params on Blackwell CUTLASS)
    modules_to_not_convert = sorted(set(all_skipped_modules))
    logger.info(f"modules_to_not_convert: {len(modules_to_not_convert)} modules")
    orig_config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [BLOCK_SIZE, BLOCK_SIZE],
        "modules_to_not_convert": modules_to_not_convert,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(orig_config, f, indent=2)

    # Copy tokenizer + extras
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    tok.save_pretrained(str(output_dir))

    for fname in ["preprocessor_config.json", "generation_config.json",
                   "chat_template.jinja", "vocab.json"]:
        src = snapshot / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    out_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1024**3

    print(f"\n{'='*60}")
    print(f"  Model:       {model_id}")
    print(f"  Output:      {output_dir}")
    print(f"  Size:        {out_size:.1f} GB")
    print(f"  Quant time:  {quant_time:.1f}s")
    print(f"  Format:      native fp8 (quant_method=fp8, block_size={BLOCK_SIZE})")
    print(f"  Quantized:   {total_quantized} layers")
    print(f"  Skipped:     {total_skipped} params")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
