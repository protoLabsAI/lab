"""Core Pydantic models shared across the lab."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GPUConfig(str, Enum):
    """GPU deployment configuration."""
    SINGLE_0 = "gpu0"
    SINGLE_1 = "gpu1"
    DUAL_TP2 = "tp2"


class ModelSpec(BaseModel):
    """Specification for a locally-served model."""
    name: str                               # swap config name e.g. "qwen-27b-int4"
    repo_id: str                            # HuggingFace ID e.g. "Qwen/Qwen3.5-27B-GPTQ-Int4"
    architecture: str = "dense"             # "dense" or "moe"
    total_params_b: float | None = None     # total params in billions
    active_params_b: float | None = None    # active params (same as total for dense)
    quantization: str | None = None         # "gptq-int4", "awq", "fp8", None (bf16)
    disk_gb: float | None = None            # weight size on disk
    gpu_config: GPUConfig = GPUConfig.SINGLE_0
    max_context: int = 131072               # max context length
    tool_parser: str | None = None          # "qwen3_xml", "llama3_json", "hermes", etc.
    cuda_graphs: bool = True                # whether CUDA graphs work (no enforce-eager)
    notes: str = ""


class BenchmarkResult(BaseModel):
    """Result from a speed or quality benchmark."""
    model: str
    task: str
    score: float
    passed: bool = False
    tokens_per_sec: float | None = None
    wall_time_s: float | None = None
    vram_peak_gb: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregated eval result across a suite."""
    model: str
    suite: str                              # "claw", "function-call", "rag", etc.
    scores: dict[str, float]                # task/dimension -> score
    pass_rate: float = 0.0                  # fraction of tasks passed
    pass_k: int = 3                         # pass^k (default pass^3)
    avg_score: float = 0.0
    avg_time_s: float | None = None
    tokens_per_sec: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None


class TrainingConfig(BaseModel):
    """Configuration for a fine-tuning run."""
    base_model: str                         # HuggingFace model ID
    method: str = "lora"                    # "lora", "qlora", "full"
    lora_rank: int = 64
    lora_alpha: int = 128
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 4096
    dataset: str = ""                       # path or HF dataset ID
    output_dir: str = ""
    gpu_config: GPUConfig = GPUConfig.SINGLE_0
    notes: str = ""
