"""Canonical paths for the protoLabs AI node."""

from pathlib import Path

# Storage drives
MODELS_ROOT = Path("/mnt/models")
HF_HOME = MODELS_ROOT / "huggingface"
DATA_ROOT = Path("/mnt/data")
SCRATCH_ROOT = Path("/mnt/scratch")

# Working directories
OUTPUT_DIR = DATA_ROOT / "comfyui" / "output"
DATASETS_DIR = DATA_ROOT / "datasets"
CHECKPOINTS_DIR = DATA_ROOT / "checkpoints"
TRAINING_DIR = DATA_ROOT / "training"
LOGS_DIR = SCRATCH_ROOT / "logs"

# Lab repo
LAB_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
