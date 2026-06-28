#!/usr/bin/env python3
"""
Quantize Qwen3.5 models to native FP8 format (matching official Qwen approach).

Directly casts weights to float8_e4m3fn with per-block scaling,
producing the `quant_method: fp8` format that vLLM handles natively.

Usage:
  source ~/dev/quant-env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python quantize_native_fp8.py Qwen/Qwen3.5-9B
  CUDA_VISIBLE_DEVICES=0 python quantize_native_fp8.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/mnt/models/quantized")
BLOCK_SIZE = 128  # Match official Qwen FP8


def should_skip(name: str, skip_patterns: list[str]) -> bool:
    """Check if a weight should be skipped (kept in original dtype)."""
    for pattern in skip_patterns:
        if pattern in name:
            return True
    return False


def quantize_weight_blockwise(weight: torch.Tensor, block_size: int = BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to FP8 E4M3 with per-block scaling.

    Returns (quantized_weight, scale) matching Qwen's official approach.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    orig_shape = weight.shape
    # Pad to block-aligned if needed
    rows, cols = weight.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size

    if pad_rows or pad_cols:
        weight = torch.nn.functional.pad(weight.float(), (0, pad_cols, 0, pad_rows))

    # Reshape into blocks
    r, c = weight.shape
    weight_blocks = weight.reshape(r // block_size, block_size, c // block_size, block_size)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)  # [br, bc, bs, bs]

    # Compute per-block scale
    block_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    scale = block_max / fp8_max

    # Quantize
    quantized = (weight_blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # Reshape back
    quantized = quantized.permute(0, 2, 1, 3).reshape(r, c)
    scale = scale.squeeze(-1).squeeze(-1)  # [br, bc]

    # Remove padding
    quantized = quantized[:orig_shape[0], :orig_shape[1]]

    return quantized, scale.float()


def quantize_model(model_id: str, output_dir: Path):
    t0 = time.time()

    logger.info(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    load_time = time.time() - t0
    logger.info(f"Loaded in {load_time:.1f}s")

    # Build skip patterns (matching official Qwen FP8)
    # NOTE: blanket "linear_attn" keeps the WHOLE DeltaNet/SSM block in bf16. The 9B's
    # DeltaNet has more projections than the 35B (in_proj_qkv / in_proj_z / out_proj in
    # addition to in_proj_a/b/conv1d); enumerating them per-model silently FP8s the missed
    # ones and corrupts the SSM. The blanket pattern is correct for any qwen3_5 hybrid.
    skip_patterns = [
        "lm_head",
        "embed_tokens",
        "linear_attn",
        "visual",
        "mtp",
        "layernorm", "layer_norm", "norm",
    ]

    # Collect modules to not convert (for config.json)
    modules_to_not_convert = []
    for name, param in model.named_parameters():
        if should_skip(name, skip_patterns):
            module_name = name.rsplit(".", 1)[0]  # Remove .weight/.bias
            if module_name not in modules_to_not_convert:
                modules_to_not_convert.append(module_name)

    logger.info(f"Skipping {len(modules_to_not_convert)} modules, quantizing the rest to FP8")

    # Quantize
    t1 = time.time()
    state_dict = model.state_dict()
    new_state_dict = {}
    quantized_count = 0
    skipped_count = 0

    for name, tensor in state_dict.items():
        if should_skip(name, skip_patterns) or tensor.ndim < 2:
            new_state_dict[name] = tensor
            skipped_count += 1
        elif name.endswith(".weight") and tensor.ndim == 2:
            q_weight, scale = quantize_weight_blockwise(tensor.float())
            new_state_dict[name] = q_weight
            scale_name = name.replace(".weight", ".weight_scale")
            new_state_dict[scale_name] = scale
            quantized_count += 1
        elif name.endswith(".weight") and tensor.ndim == 3:
            # MoE expert weights: [num_experts, in, out] — quantize each expert
            experts = []
            scales = []
            for i in range(tensor.shape[0]):
                q_w, s = quantize_weight_blockwise(tensor[i].float())
                experts.append(q_w)
                scales.append(s)
            new_state_dict[name] = torch.stack(experts)
            scale_name = name.replace(".weight", ".weight_scale")
            new_state_dict[scale_name] = torch.stack(scales)
            quantized_count += 1
        else:
            new_state_dict[name] = tensor
            skipped_count += 1

    quant_time = time.time() - t1
    logger.info(f"Quantized {quantized_count} layers, skipped {skipped_count} in {quant_time:.1f}s")

    # Free original model + state dict to reclaim RAM before saving
    del model
    del state_dict
    import gc
    gc.collect()

    # Save
    logger.info(f"Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights in shards to stay within RAM
    # Target ~5GB per shard to keep peak memory manageable
    SHARD_MAX_BYTES = 5 * 1024**3
    shard_idx = 0
    current_shard = {}
    current_bytes = 0
    shard_files = []
    weight_map = {}

    seen_data_ptrs = {}
    for name, tensor in new_state_dict.items():
        t = tensor.contiguous()
        ptr = t.data_ptr()
        if ptr in seen_data_ptrs:
            t = t.clone()
        else:
            seen_data_ptrs[ptr] = name

        tensor_bytes = t.nelement() * t.element_size()
        if current_bytes + tensor_bytes > SHARD_MAX_BYTES and current_shard:
            # Save current shard
            shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file(current_shard, str(output_dir / shard_name))
            shard_files.append(shard_name)
            for k in current_shard:
                weight_map[k] = shard_name
            current_shard = {}
            current_bytes = 0
            shard_idx += 1

        current_shard[name] = t
        current_bytes += tensor_bytes

    # Save last shard
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(current_shard, str(output_dir / shard_name))
        shard_files.append(shard_name)
        for k in current_shard:
            weight_map[k] = shard_name
        del current_shard

    # Rename with final count
    total_shards = len(shard_files)
    for i, old_name in enumerate(shard_files):
        new_name = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        (output_dir / old_name).rename(output_dir / new_name)
        for k in weight_map:
            if weight_map[k] == old_name:
                weight_map[k] = new_name

    # Write index
    total_size = sum(t.nelement() * t.element_size() for t in new_state_dict.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    del new_state_dict

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Build config with quantization metadata
    orig_config = json.load(open(
        Path(os.environ["HF_HOME"]) / "hub" /
        f"models--{model_id.replace('/', '--')}" / "snapshots" /
        os.listdir(Path(os.environ["HF_HOME"]) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots")[0] /
        "config.json"
    ))

    orig_config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_per_tensor": False,
        "act_per_tensor": False,
        "weight_block_size": [BLOCK_SIZE, BLOCK_SIZE],
        "modules_to_not_convert": sorted(modules_to_not_convert),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(orig_config, f, indent=2)

    # Copy additional config files from original
    orig_snapshot = (
        Path(os.environ["HF_HOME"]) / "hub" /
        f"models--{model_id.replace('/', '--')}" / "snapshots" /
        os.listdir(Path(os.environ["HF_HOME"]) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots")[0]
    )
    for fname in ["preprocessor_config.json", "video_preprocessor_config.json",
                   "generation_config.json", "chat_template.jinja", "vocab.json"]:
        src = orig_snapshot / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Size
    total = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    size_gb = total / 1024**3
    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Model:       {model_id}")
    print(f"  Output:      {output_dir}")
    print(f"  Size:        {size_gb:.1f} GB")
    print(f"  Quant time:  {quant_time:.1f}s")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  Format:      native fp8 (quant_method=fp8, block_size={BLOCK_SIZE})")
    print(f"  Quantized:   {quantized_count} layers")
    print(f"  Skipped:     {skipped_count} params (Mamba, embed, norms)")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python quantize_native_fp8.py <model_id> [output_dir]")
        sys.exit(1)

    model_id = sys.argv[1]
    model_short = model_id.split("/")[-1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_ROOT / f"{model_short}-FP8"

    quantize_model(model_id, output_dir)


if __name__ == "__main__":
    main()
