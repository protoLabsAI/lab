"""lab-core — shared Pydantic models and utilities for protoLabs AI Lab."""

from lab_core.models import (
    BenchmarkResult,
    EvalResult,
    GPUConfig,
    ModelSpec,
    TrainingConfig,
)
from lab_core.paths import (
    DATA_ROOT,
    DATASETS_DIR,
    HF_HOME,
    LOGS_DIR,
    MODELS_ROOT,
    OUTPUT_DIR,
    SCRATCH_ROOT,
)

__all__ = [
    "BenchmarkResult",
    "EvalResult",
    "GPUConfig",
    "ModelSpec",
    "TrainingConfig",
    "DATA_ROOT",
    "DATASETS_DIR",
    "HF_HOME",
    "LOGS_DIR",
    "MODELS_ROOT",
    "OUTPUT_DIR",
    "SCRATCH_ROOT",
]
