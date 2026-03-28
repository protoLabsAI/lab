# TurboQuant KV Cache Compression — Experiment Results

KV cache quantization using [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) fork
on 2x NVIDIA RTX PRO 6000 Blackwell (SM 12.0, 96GB each).

## Key Finding

TurboQuant works on SM 12.0 (GB202 Blackwell) — not just SM 12.1 (GB10) as the fork claims.
The shared memory budget is identical (101,376 bytes opt-in). Required patching 6 SM capability
checks from `== (12, 1)` to `>= (12, 0)`.

## Results

### Single GPU Benchmarks (turboquant35 = 3.5-bit KV cache)

| Model | Standard tok/s | + TurboQuant tok/s | KV Cache Tokens | Max 262K Concurrent |
|-------|:-:|:-:|:-:|:-:|
| **35B MoE FP8** | 180 | **68** | **2,464,000** | **35x** |
| **27B FP8** | 36 | **23** | **876,128** | 24x (131K) |
| **9B bf16** | 92 | **54** | **1,971,200** | 107x (65K) |

### What TurboQuant Does

- Compresses KV cache from FP16 (16-bit) to 3.5-bit effective → **~4.5x compression**
- Trades decode speed for massive KV cache capacity
- Quality verified: coherent thinking, correct output on all models tested

### Best Use Case

**35B MoE FP8 + TurboQuant** on single GPU:
- 68 tok/s (still fast for MoE)
- 262K context
- 35 concurrent full-context sessions
- 2.46M total KV cache tokens
- Leaves GPU 1 completely free

### What Doesn't Work (Yet)

- **TP=2 (tensor parallel)**: Fork doesn't support multi-GPU. 122B at 262K is blocked.
- **CUDA graphs**: Not compatible with TurboQuant path yet.
- **Proper calibration**: We used synthetic (uniform) outlier metadata. Real calibration
  would run activation stats on representative prompts to select true high-variance channels.
  Quality would improve with proper calibration.

## Setup

### Environment

```bash
# Separate venv from production vllm-env
~/dev/tq-env/          # vLLM TurboQuant fork (SM 12.0 patched)
~/dev/vllm-turboquant/ # Fork source with SM patches
```

### SM 12.0 Patches Applied

Files modified in `~/dev/vllm-turboquant/`:
- `vllm/v1/attention/backends/triton_attn.py` — `GB10_CAPABILITY = DeviceCapability(12, 0)`
- `vllm/v1/attention/ops/triton_turboquant_decode.py` — `capability < (12, 0)`
- `vllm/v1/attention/ops/triton_turboquant_kv_update.py` — `capability < (12, 0)`
- `vllm/v1/attention/ops/triton_prefill_attention.py` — `capability >= (12, 0)`
- `vllm/config/cache.py` — `capability < (12, 0)`

### Metadata Files

```bash
/tmp/turboquant_9b.json    # 8 attention layers
/tmp/turboquant_27b.json   # 16 attention layers
/tmp/turboquant_35b.json   # 10 attention layers
/tmp/turboquant_122b.json  # 12 attention layers (untested, needs TP=2)
```

### Running

```bash
source ~/dev/tq-env/bin/activate

# 35B MoE FP8 + TurboQuant (best single-GPU config)
CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/models/huggingface \
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --served-model-name local \
    --gpu-memory-utilization 0.90 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-chunked-prefill \
    --kv-cache-dtype turboquant35 \
    --enable-turboquant \
    --turboquant-metadata-path /tmp/turboquant_35b.json
```

## Next Steps

1. **Proper calibration** — Run activation stats on real prompts instead of uniform synthetic metadata
2. **TP=2 support** — Watch for upstream fork updates enabling multi-GPU
3. **RotorQuant** — 10-19x faster rotation stage, no SM restriction. Monitor [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) for vLLM integration
4. **Benchmark under load** — Test with concurrent users to validate the 35x concurrency claim
