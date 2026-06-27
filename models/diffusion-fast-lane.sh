#!/bin/bash
# Fast lane = DiffusionGemma 26B-A4B (text-diffusion) FP8, GPU 1, :8002, 256K.
# Runs the day-zero diffusion runner via Google's vLLM container (not in pip vLLM 0.22.1).
# FP8 (~26GB) co-resides with Fish TTS + embed. util 0.66 (was 0.68, was 0.72): embed-server
# grew ~1.2GB->6.4GB, so 0.68 only fit via boot ordering and OOMed by ~167MB at graph capture on
# any standalone restart. 0.66 restores restartability while keeping 256K (verified 2026-06-14).
# canvas_length 256; gemma4 tool/reasoning parsers (DG is Gemma-4 based). Tool calling untested
# but harmless; judges must NOT route here (DG can't guided-decode — relocated separately).
set -euo pipefail
NAME=diffusiongemma-fast
/usr/bin/docker rm -f "$NAME" >/dev/null 2>&1 || true
exec /usr/bin/docker run --rm --name "$NAME" --ipc=host --gpus '"device=1"' \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /mnt/models/huggingface:/root/.cache/huggingface \
  -v /home/ava/dev/lab/models/diffusion-fast-nothink.jinja:/tmp/nothink.jinja:ro \
  -p 8002:8002 \
  vllm/vllm-openai:gemma \
    --model google/diffusiongemma-26B-A4B-it \
    --quantization fp8 \
    --served-model-name local-fast \
    --chat-template /tmp/nothink.jinja \
    --override-generation-config '{"max_new_tokens": 4096}' \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.66 \
    --diffusion-config '{"canvas_length":256}' \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --host 0.0.0.0 --port 8002
