# MoE Kernel Tuning for Blackwell

## What and Why

vLLM ships pre-tuned Triton fused MoE kernel configs for common GPUs (A100, H100, A10G).
For unknown GPUs — including the RTX PRO 6000 Blackwell — it falls back to a generic config
and logs a warning at startup:

```
Config file not found at vllm/model_executor/layers/fused_moe/configs/
E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json
```

Without the config, the fused MoE dispatch kernel uses suboptimal block sizes, warp counts,
and split-k factors. Tuning generates a GPU-specific config that can significantly improve
MoE expert routing throughput.

## When to Tune

- Any new MoE model deployed on this machine
- After a major vLLM upgrade that changes the kernel
- The warning above appears in vLLM logs at startup

Dense models (e.g. Qwen3.6-27B) do not have MoE layers — no tuning needed.

## How to Tune

### Prerequisites

1. The target GPU must be free (no other vLLM instance using it)
2. Use the vllm-env Python environment
3. Set `HF_HOME` so the model can be loaded

### Command

```bash
# Stop any vLLM instance on the target GPU first
sudo systemctl stop vllm   # frees GPU 0
# or: sudo systemctl stop vllm-voice  # frees GPU 1

# Run tuning in a persistent tmux session
tmux new-session -d -s moe-tune -x 220 -y 50 \
  "CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/models/huggingface \
   /home/ava/dev/vllm-env/bin/python \
   /home/ava/dev/vllm-build/benchmarks/kernels/benchmark_moe.py \
   --model <model-repo-id> \
   --tp-size 1 \
   --tune \
   --dtype fp8_w8a8 \
   --save-dir /home/ava/dev/lab/models/moe-configs/ \
   2>&1 | tee /mnt/scratch/logs/moe-tune-<model>.log"
```

**Use tmux** — tuning takes 1-3 hours and must survive terminal disconnects.

### Monitor Progress

```bash
# Watch log
tail -f /mnt/scratch/logs/moe-tune-<model>.log | grep -v raylet

# Check completed batch sizes
grep "Completed tuning for batch_size" /mnt/scratch/logs/moe-tune-<model>.log

# Find output JSON when done
grep "Writing best config" /mnt/scratch/logs/moe-tune-<model>.log
```

### Expected Output

The script tunes across batch sizes: 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
1024, 1536, 2048, 3072, 4096 — each takes ~7-10 minutes. Total runtime ~2-3 hours.

When complete, it writes:
```
Writing best config to ./E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_...,dtype=fp8_w8a8,...json
```

### Install the Config

Use the install script — it symlinks every config in `models/moe-configs/` into the vLLM venv's `fused_moe/configs/` directory. Symlinks survive in-place edits (re-tuning a config is automatically picked up); a pip-driven vLLM upgrade may clobber the configs/ dir, in which case re-run this script.

```bash
bash models/install-moe-configs.sh
sudo systemctl restart vllm-fast  # or vllm, depending on which service serves the tuned model
```

After restart, verify in the vLLM log:
- ❌ Old (config missing): `WARNING ... Using default MoE config. Performance might be sub-optimal!`
- ✅ New (config loaded): `INFO ... Using configuration from .../fused_moe/configs/E=256,N=512,...json for MoE layer.`

## Qwen3.6-35B-A3B-FP8 Tuning (April 2026)

**Model:** `Qwen/Qwen3.6-35B-A3B-FP8`
**GPU:** NVIDIA RTX PRO 6000 Blackwell (GPU 0, 96 GB)
**Date:** 2026-04-25
**Duration:** ~2.5 hours (17:07 – 19:30)
**Batch sizes completed:** 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
**Configurations tested:** 640 per batch size
**dtype:** fp8_w8a8, block_shape=[128, 128]

**MoE config parameters for this model:**
- Experts (E): 256 (128 active per token)
- Intermediate size (N): 512 (shard intermediate size // 2)
- dtype: fp8_w8a8

**Notes:**
- vLLM was stopped on GPU 0 during tuning to free full 94 GB for profiling
- Ray raylet warnings about `/tmp` disk space are benign (OS drive is 96% full on `/tmp` symlink to 1TB NVMe)
- Config written to `--save-dir ./` (default), then copied to vLLM configs directory
- Restart `vllm-voice` after installing to verify warning is gone

## Config File Location

```
/home/ava/dev/vllm-build/vllm/model_executor/layers/fused_moe/configs/
  E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json
```

## Related

- vLLM swap script: `models/vllm-swap.sh` (`qwen-36b-voice` profile)
- Service file: `/etc/systemd/system/vllm-voice.service`
- Tuning script: `/home/ava/dev/vllm-build/benchmarks/kernels/benchmark_moe.py`
