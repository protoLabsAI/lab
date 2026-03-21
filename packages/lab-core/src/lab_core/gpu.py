"""GPU utilities for the protoLabs AI node."""

from __future__ import annotations

import subprocess
import json


def get_gpu_info() -> list[dict]:
    """Get GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "power_draw_w": float(parts[4]),
                    "power_limit_w": float(parts[5]),
                    "temp_c": float(parts[6]),
                })
        return gpus
    except Exception:
        return []


def get_vram_free_gb(device: int = 0) -> float:
    """Get free VRAM in GB for a device."""
    gpus = get_gpu_info()
    for gpu in gpus:
        if gpu["index"] == device:
            return (gpu["memory_total_mb"] - gpu["memory_used_mb"]) / 1024
    return 0.0
