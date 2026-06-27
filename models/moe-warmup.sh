#!/usr/bin/env bash
# Warm up fused-MoE / Triton / CUDA-graph kernel shapes after a vLLM replica
# starts, so production traffic doesn't eat first-request JIT compilation spikes
# (jit_monitor warns: "fused_moe_kernel JIT compilation during inference ...").
# Covers: prefill prompt lengths, decode batch sizes, guided/JSON, tool calls.
#
# Usage: moe-warmup.sh [port]   (default 8000). Wired as ExecStartPost on the
# vllm replica units; idempotent + safe to run by hand against a live replica.
set -uo pipefail
PORT="${1:-8000}"
URL="http://localhost:${PORT}/v1/chat/completions"
post() { curl -s --max-time 180 "$URL" -H "Content-Type: application/json" -d "$1" >/dev/null 2>&1; }

# wait until the replica is serving (ExecStartPost can fire before the model loads)
for i in $(seq 1 90); do
  curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"local"' && break
  sleep 5
done

# 1) prefill shapes — varied prompt lengths bucket the fused_moe M dimension
for n in 16 128 512 2048 8192 32768; do
  pad=$(head -c $((n*4)) /dev/zero 2>/dev/null | tr '\0' 'a')
  post "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize briefly: ${pad}\"}],\"max_tokens\":16,\"temperature\":0}"
done

# 2) decode batch shapes — concurrency sweep warms the decode-M graphs/kernels
for c in 1 2 4 8 16 32; do
  for _ in $(seq 1 "$c"); do
    post "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly to thirty.\"}],\"max_tokens\":96,\"temperature\":0}" &
  done
  wait
done

# 3) guided/structured (apply_token_bitmask_inplace_kernel)
post "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Return ONLY JSON: {\\\"x\\\":1}\"}],\"max_tokens\":64,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}}"

# 4) tool-call shape
post "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}],\"tool_choice\":\"auto\",\"max_tokens\":64}"

echo "moe-warmup: done for :${PORT}"
