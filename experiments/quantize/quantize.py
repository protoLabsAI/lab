#!/usr/bin/env python3
"""
protoLabs Model Quantization Pipeline

Quantize HuggingFace models for vLLM serving using llm-compressor.

Methods:
  fp8       FP8_DYNAMIC W8A8 — no calibration, ~50% size, ~99% quality, minutes
  gptq      W4A16 GPTQ INT4 — calibration required, ~75% size, ~95% quality, hours
  awq       AWQ INT4 — calibration required, ~75% size, ~95% quality, fast

Usage:
  # FP8 (fast, no calibration)
  python quantize.py --model Qwen/Qwen3.5-9B --method fp8

  # GPTQ INT4 (slower, needs calibration data)
  python quantize.py --model Qwen/Qwen3.5-9B --method gptq --calib-samples 512

  # AWQ INT4
  python quantize.py --model Qwen/Qwen3.5-9B --method awq --calib-samples 128

  # Custom output dir
  python quantize.py --model Qwen/Qwen3.5-9B --method fp8 --output /mnt/models/custom-quants/9B-FP8

  # Custom calibration dataset (for agentic/tool-calling tuned quants)
  python quantize.py --model Qwen/Qwen3.5-9B --method gptq --calib-dataset path/to/data.jsonl

Environment:
  Requires ~/dev/vllm-env (has compressed_tensors, transformers, torch).
  Install llmcompressor: pip install llmcompressor
  For AWQ: pip install autoawq
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("/mnt/models/quantized")
DEFAULT_CALIB_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_CALIB_SPLIT = "train_sft[:512]"


# ---------------------------------------------------------------------------
# FP8 Dynamic (llm-compressor)
# ---------------------------------------------------------------------------

def quantize_fp8(
    model_id: str,
    output_dir: Path,
    trust_remote_code: bool = True,
) -> dict:
    """FP8_DYNAMIC W8A8 quantization. No calibration needed."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    logger.info(f"Loading {model_id} for FP8 quantization...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Log VRAM after load
    vram = _get_gpu_memory()
    logger.info(f"VRAM after load: {vram}")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    logger.info("Running FP8_DYNAMIC quantization...")
    t1 = time.time()
    oneshot(model=model, recipe=recipe)
    quant_time = time.time() - t1
    logger.info(f"Quantization completed in {quant_time:.1f}s")

    # Save
    logger.info(f"Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    output_size = _dir_size_gb(output_dir)
    logger.info(f"Output size: {output_size:.1f} GB")

    return {
        "method": "fp8_dynamic",
        "model": model_id,
        "output_dir": str(output_dir),
        "load_time_s": round(load_time, 1),
        "quant_time_s": round(quant_time, 1),
        "total_time_s": round(time.time() - t0, 1),
        "output_size_gb": round(output_size, 1),
        "vram_after_load": vram,
    }


# ---------------------------------------------------------------------------
# GPTQ W4A16 (llm-compressor)
# ---------------------------------------------------------------------------

def quantize_gptq(
    model_id: str,
    output_dir: Path,
    calib_dataset: str = DEFAULT_CALIB_DATASET,
    calib_split: str = DEFAULT_CALIB_SPLIT,
    calib_samples: int = 512,
    max_seq_length: int = 2048,
    trust_remote_code: bool = True,
) -> dict:
    """W4A16 GPTQ quantization via llm-compressor. Requires calibration data."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    logger.info(f"Loading {model_id} for GPTQ quantization...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    vram = _get_gpu_memory()
    logger.info(f"VRAM after load: {vram}")

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    # Resolve calibration dataset
    dataset_kwargs = _resolve_calib_dataset(calib_dataset, calib_split)

    logger.info(
        f"Running GPTQ W4A16 quantization "
        f"({calib_samples} samples, seq_len={max_seq_length})..."
    )
    t1 = time.time()
    oneshot(
        model=model,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=calib_samples,
        **dataset_kwargs,
    )
    quant_time = time.time() - t1
    logger.info(f"Quantization completed in {quant_time:.1f}s")

    logger.info(f"Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    output_size = _dir_size_gb(output_dir)
    logger.info(f"Output size: {output_size:.1f} GB")

    return {
        "method": "gptq_w4a16",
        "model": model_id,
        "output_dir": str(output_dir),
        "calib_dataset": calib_dataset,
        "calib_samples": calib_samples,
        "load_time_s": round(load_time, 1),
        "quant_time_s": round(quant_time, 1),
        "total_time_s": round(time.time() - t0, 1),
        "output_size_gb": round(output_size, 1),
        "vram_after_load": vram,
    }


# ---------------------------------------------------------------------------
# AWQ INT4 (autoawq)
# ---------------------------------------------------------------------------

def quantize_awq(
    model_id: str,
    output_dir: Path,
    calib_samples: int = 128,
    trust_remote_code: bool = True,
) -> dict:
    """AWQ INT4 quantization via AutoAWQ."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"Loading {model_id} for AWQ quantization...")
    t0 = time.time()

    model = AutoAWQForCausalLM.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    vram = _get_gpu_memory()
    logger.info(f"VRAM after load: {vram}")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    logger.info(f"Running AWQ INT4 quantization ({calib_samples} samples)...")
    t1 = time.time()
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        max_calib_samples=calib_samples,
    )
    quant_time = time.time() - t1
    logger.info(f"Quantization completed in {quant_time:.1f}s")

    logger.info(f"Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir), safetensors=True, shard_size="4GB")
    tokenizer.save_pretrained(output_dir)

    output_size = _dir_size_gb(output_dir)
    logger.info(f"Output size: {output_size:.1f} GB")

    return {
        "method": "awq_int4",
        "model": model_id,
        "output_dir": str(output_dir),
        "calib_samples": calib_samples,
        "load_time_s": round(load_time, 1),
        "quant_time_s": round(quant_time, 1),
        "total_time_s": round(time.time() - t0, 1),
        "output_size_gb": round(output_size, 1),
        "vram_after_load": vram,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_calib_dataset(dataset: str, split: str) -> dict:
    """Resolve calibration dataset — local JSONL or HuggingFace dataset."""
    path = Path(dataset)
    if path.exists() and path.suffix in (".jsonl", ".json"):
        logger.info(f"Using local calibration dataset: {path}")
        return {"dataset": str(path), "splits": {"calibration": "train"}}
    else:
        logger.info(f"Using HuggingFace dataset: {dataset} [{split}]")
        return {"dataset": dataset, "splits": {"calibration": split}}


def _get_gpu_memory() -> dict:
    """Get current GPU memory usage."""
    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        info[f"gpu_{i}"] = f"{allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return info


def _dir_size_gb(path: Path) -> float:
    """Get directory size in GB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024**3


def _default_output_name(model_id: str, method: str) -> str:
    """Generate output dir name from model and method."""
    name = model_id.split("/")[-1]
    suffix = {"fp8": "FP8", "gptq": "GPTQ-Int4", "awq": "AWQ-Int4"}[method]
    return f"{name}-{suffix}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantize models for vLLM serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FP8 — fast, no calibration
  python quantize.py --model Qwen/Qwen3.5-9B --method fp8

  # GPTQ INT4 — slower, calibrated
  python quantize.py --model Qwen/Qwen3.5-9B --method gptq

  # AWQ INT4 — fast, calibrated
  python quantize.py --model Qwen/Qwen3.5-9B --method awq

  # Custom calibration data (e.g. agentic/tool-calling)
  python quantize.py --model Qwen/Qwen3.5-9B --method gptq \\
      --calib-dataset training/data/my_calib.jsonl

  # Push to HuggingFace Hub
  python quantize.py --model Qwen/Qwen3.5-9B --method fp8 --push-to-hub
        """,
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["fp8", "gptq", "awq"],
        help="Quantization method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}/<model>-<method>)",
    )
    parser.add_argument(
        "--calib-dataset",
        default=DEFAULT_CALIB_DATASET,
        help="Calibration dataset (HF name or local JSONL path)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=None,
        help="Number of calibration samples (default: 512 for GPTQ, 128 for AWQ)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length for calibration (default: 2048)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push quantized model to HuggingFace Hub after quantization",
    )
    args = parser.parse_args()

    # Resolve output dir
    if args.output is None:
        args.output = DEFAULT_OUTPUT_ROOT / _default_output_name(
            args.model, args.method
        )

    logger.info(f"Quantization plan:")
    logger.info(f"  Model:    {args.model}")
    logger.info(f"  Method:   {args.method}")
    logger.info(f"  Output:   {args.output}")
    if args.method in ("gptq", "awq"):
        logger.info(f"  Dataset:  {args.calib_dataset}")
    logger.info(f"  GPUs:     {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
    logger.info("")

    # Run quantization
    if args.method == "fp8":
        result = quantize_fp8(args.model, args.output)

    elif args.method == "gptq":
        samples = args.calib_samples or 512
        result = quantize_gptq(
            args.model,
            args.output,
            calib_dataset=args.calib_dataset,
            calib_samples=samples,
            max_seq_length=args.max_seq_length,
        )

    elif args.method == "awq":
        samples = args.calib_samples or 128
        result = quantize_awq(args.model, args.output, calib_samples=samples)

    # Save run metadata
    result_path = args.output / "quantize_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Run metadata saved to {result_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Method:      {result['method']}")
    print(f"  Model:       {result['model']}")
    print(f"  Output:      {result['output_dir']}")
    print(f"  Output size: {result['output_size_gb']} GB")
    print(f"  Quant time:  {result['quant_time_s']}s")
    print(f"  Total time:  {result['total_time_s']}s")
    print("=" * 60)

    # Push to hub
    if args.push_to_hub:
        _push_to_hub(args.output, args.model, args.method)

    # Suggest next steps
    print(f"\nNext steps:")
    print(f"  # Serve with vLLM")
    print(f"  vllm serve {args.output} --host 0.0.0.0 --port 8000")
    print(f"")
    print(f"  # Run eval to quality-gate")
    print(f"  cd ~/dev/lab/evals && ./run.sh profile --name quick --model local")


def _push_to_hub(output_dir: Path, model_id: str, method: str):
    """Push quantized model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    repo_name = _default_output_name(model_id, method)
    org = "ArtificialCitizens"
    repo_id = f"{org}/{repo_name}"

    logger.info(f"Pushing to {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
