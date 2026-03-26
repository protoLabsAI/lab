"""Utilities for the vLLM management dashboard."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

VLLM_SWAP_SH = Path(__file__).parent.parent.parent / "models" / "vllm-swap.sh"
VLLM_LOG = Path("/mnt/scratch/logs/vllm-swap.log")
VLLM_BASE_URL = "http://localhost:8000"


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    power_limit_w: float
    temp_c: float

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024

    @property
    def memory_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0
        return self.memory_used_mb / self.memory_total_mb * 100


def get_gpu_info() -> list[GPUInfo]:
    """Get GPU status via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1].replace("NVIDIA ", ""),
                    memory_used_mb=float(parts[2]),
                    memory_total_mb=float(parts[3]),
                    power_draw_w=float(parts[4]),
                    power_limit_w=float(parts[5]),
                    temp_c=float(parts[6]),
                ))
        return gpus
    except Exception:
        return []


def get_gpu_processes() -> list[dict]:
    """Get processes using the GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        procs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                procs.append({
                    "pid": parts[0],
                    "gpu_uuid": parts[1],
                    "memory_mb": parts[2],
                    "name": parts[3].split("/")[-1][:40],
                })
        return procs
    except Exception:
        return []


# ── Model configs from vllm-swap.sh ──────────────────────────────────

@dataclass
class SwapConfig:
    name: str
    category: str  # production, training, creative, tp2


def parse_swap_configs() -> list[SwapConfig]:
    """Parse available model configs from vllm-swap.sh."""
    configs = []
    # Categories based on the script's structure
    production = [
        "qwen-27b-int4", "qwen-27b-int4-mtp", "qwen-35b",
        "qwen-9b-mtp", "qwen-4b-int4",
    ]
    training = ["qwen-9b", "qwen-4b", "qwen-2b", "qwen-0.8b"]
    creative = ["cydonia-24b", "llama-70b"]
    tp2 = ["qwen-122b-int4", "qwen-35b-tp2", "qwen-27b-int4-tp2"]

    for name in production:
        configs.append(SwapConfig(name=name, category="Production"))
    for name in training:
        configs.append(SwapConfig(name=name, category="Training"))
    for name in creative:
        configs.append(SwapConfig(name=name, category="Creative"))
    for name in tp2:
        configs.append(SwapConfig(name=name, category="TP=2 (Both GPUs)"))
    return configs


def get_config_choices() -> list[str]:
    """Get dropdown choices grouped by category."""
    configs = parse_swap_configs()
    choices = []
    current_cat = None
    for c in configs:
        if c.category != current_cat:
            current_cat = c.category
        choices.append(f"[{c.category}] {c.name}")
    return choices


def extract_config_name(choice: str) -> str:
    """Extract config name from dropdown choice string."""
    match = re.search(r'\] (.+)$', choice)
    return match.group(1) if match else choice


# ── vLLM status ──────────────────────────────────────────────────────

def get_vllm_status() -> dict:
    """Get current vLLM model status."""
    status = {
        "running": False,
        "model": "—",
        "max_context": "—",
        "requests_running": 0,
        "requests_waiting": 0,
        "kv_cache_pct": 0.0,
    }
    try:
        r = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            if models:
                status["running"] = True
                status["model"] = models[0].get("id", "unknown")
    except Exception:
        pass

    # Parse metrics
    try:
        r = requests.get(f"{VLLM_BASE_URL}/metrics", timeout=3)
        if r.status_code == 200:
            text = r.text
            for line in text.split("\n"):
                if line.startswith("vllm:num_requests_running{"):
                    status["requests_running"] = int(float(line.split()[-1]))
                elif line.startswith("vllm:num_requests_waiting{"):
                    status["requests_waiting"] = int(float(line.split()[-1]))
                elif line.startswith("vllm:gpu_cache_usage_perc{"):
                    status["kv_cache_pct"] = float(line.split()[-1]) * 100
    except Exception:
        pass

    return status


# ── Model swap ───────────────────────────────────────────────────────

def swap_model(config_name: str) -> str:
    """Swap to a new model. Returns log output."""
    try:
        result = subprocess.run(
            ["bash", str(VLLM_SWAP_SH), config_name],
            capture_output=True, text=True, timeout=360,
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            return f"Swapped to {config_name}\n\n{output}"
        else:
            return f"FAILED (exit {result.returncode})\n\n{output}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT — swap to {config_name} took > 6 minutes"
    except Exception as e:
        return f"ERROR: {e}"


# ── Log tailing ──────────────────────────────────────────────────────

def tail_log(n: int = 50) -> str:
    """Tail the vLLM swap log."""
    try:
        if VLLM_LOG.exists():
            result = subprocess.run(
                ["tail", "-n", str(n), str(VLLM_LOG)],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout
        return "No log file found."
    except Exception as e:
        return f"Error reading log: {e}"


# ── Chat ─────────────────────────────────────────────────────────────

def chat_completion(message: str, history: list, max_tokens: int = 500) -> str:
    """Send a chat completion request to vLLM."""
    messages = []
    for h in history:
        if isinstance(h, dict):
            messages.append(h)
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})

    messages.append({"role": "user", "content": message})

    try:
        r = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": "local",
                "messages": messages,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            return content or "(empty response)"
        else:
            return f"Error {r.status_code}: {r.text[:200]}"
    except requests.ConnectionError:
        return "vLLM is not running. Swap to a model first."
    except Exception as e:
        return f"Error: {e}"
