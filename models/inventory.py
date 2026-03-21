"""Model inventory — registry of all models on disk and their configs."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from lab_core.models import ModelSpec, GPUConfig

# ── Model Registry ──────────────────────────────────────────────────────────

MODELS: list[ModelSpec] = [
    # Single GPU — daily drivers
    ModelSpec(
        name="qwen-27b-int4",
        repo_id="Qwen/Qwen3.5-27B-GPTQ-Int4",
        total_params_b=27, active_params_b=27,
        quantization="gptq-int4", disk_gb=14,
        max_context=131072, tool_parser="qwen3_xml",
        notes="Daily driver. Best all-rounder. 44 tok/s with CUDA graphs.",
    ),
    ModelSpec(
        name="qwen-27b",
        repo_id="Qwen/Qwen3.5-27B",
        total_params_b=27, active_params_b=27,
        disk_gb=52, max_context=131072, tool_parser="qwen3_xml",
        notes="BF16 baseline. Same quality as INT4, 3.7x larger.",
    ),
    ModelSpec(
        name="qwen-35b",
        repo_id="Qwen/Qwen3.5-35B-A3B",
        architecture="moe", total_params_b=35, active_params_b=3,
        disk_gb=72, max_context=65536, tool_parser="qwen3_xml",
        notes="Speed king. 170 tok/s with CUDA graphs. BF16 only (INT4 unstable on MoE).",
    ),
    ModelSpec(
        name="qwen-122b-int4-1gpu",
        repo_id="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        architecture="moe", total_params_b=122, active_params_b=10,
        quantization="gptq-int4", disk_gb=35,
        max_context=65536, tool_parser="qwen3_xml",
        cuda_graphs=False,
        notes="Highest quality ceiling. enforce-eager only (CUDA graphs crash on large MoE).",
    ),
    ModelSpec(
        name="llama-70b",
        repo_id="casperhansen/llama-3.3-70b-instruct-awq",
        total_params_b=70, active_params_b=70,
        quantization="awq", disk_gb=40,
        max_context=131072, tool_parser="llama3_json",
        notes="Creative writing + roleplay specialist. Weak on agentic tasks.",
    ),
    ModelSpec(
        name="omnicoder",
        repo_id="Tesslate/OmniCoder-9B",
        total_params_b=9, active_params_b=9,
        disk_gb=18, max_context=262144, tool_parser="hermes",
        notes="Fine-tune candidate. 92 tok/s. Strong research synthesis.",
    ),

    # Dual GPU (TP=2) — needs both cards
    ModelSpec(
        name="qwen-35b-tp2",
        repo_id="Qwen/Qwen3.5-35B-A3B",
        architecture="moe", total_params_b=35, active_params_b=3,
        disk_gb=72, max_context=253952, tool_parser="qwen3_xml",
        gpu_config=GPUConfig.DUAL_TP2,
        notes="Best config: 3/4 pass^3, 170 tok/s, 250K ctx. Needs both GPUs.",
    ),
    ModelSpec(
        name="qwen-122b-int4",
        repo_id="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        architecture="moe", total_params_b=122, active_params_b=10,
        quantization="gptq-int4", disk_gb=35,
        max_context=131072, tool_parser="qwen3_xml",
        gpu_config=GPUConfig.DUAL_TP2,
        cuda_graphs=False,
        notes="TP=2 with enforce-eager. 128K context.",
    ),
    ModelSpec(
        name="qwen-122b",
        repo_id="Qwen/Qwen3.5-122B-A10B-FP8",
        architecture="moe", total_params_b=122, active_params_b=10,
        quantization="fp8", disk_gb=70,
        max_context=65536, tool_parser="qwen3_xml",
        gpu_config=GPUConfig.DUAL_TP2,
        cuda_graphs=False,
        notes="Original FP8. 64K context at TP=2.",
    ),
]


@click.command()
@click.option("--gpu", type=click.Choice(["single", "dual", "all"]), default="all")
def main(gpu: str):
    """Show model inventory."""
    console = Console()
    table = Table(title="protoLabs Model Inventory")
    table.add_column("Name", style="bold cyan")
    table.add_column("Params", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("Quant")
    table.add_column("Disk", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("GPU")
    table.add_column("Notes", max_width=50)

    for m in MODELS:
        if gpu == "single" and m.gpu_config == GPUConfig.DUAL_TP2:
            continue
        if gpu == "dual" and m.gpu_config != GPUConfig.DUAL_TP2:
            continue

        table.add_row(
            m.name,
            f"{m.total_params_b}B" if m.total_params_b else "—",
            f"{m.active_params_b}B" if m.active_params_b else "—",
            m.quantization or "bf16",
            f"{m.disk_gb}GB" if m.disk_gb else "—",
            f"{m.max_context // 1024}K",
            m.gpu_config.value,
            m.notes[:50],
        )

    console.print(table)


if __name__ == "__main__":
    main()
