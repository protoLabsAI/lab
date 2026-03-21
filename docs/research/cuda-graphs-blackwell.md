# CUDA Graphs on Blackwell SM 12.0

## Summary

CUDA graphs work on NVIDIA Blackwell GPUs (SM 12.0) for single-GPU, single-user workloads.
The speedup is dramatic — especially for MoE models where routing overhead was the bottleneck.

## Test Results (March 2026)

| Model | enforce-eager | CUDA graphs | Speedup |
|-------|:---:|:---:|:---:|
| Qwen 35B MoE (3B active) | 30 tok/s | **170 tok/s** | **5.7x** |
| Qwen 35B MoE INT4 (3B active) | ~30 tok/s | **200 tok/s** | **6.7x** |
| OmniCoder 9B (dense) | 30 tok/s | **92 tok/s** | **3.1x** |
| Qwen 27B INT4 (dense) | 30 tok/s | **44 tok/s** | **1.5x** |
| Llama 70B AWQ (dense) | 30 tok/s | **38 tok/s** | **1.3x** |
| Qwen 122B MoE INT4 (10B active) | 30 tok/s | **CRASH** | — |

## Why MoE Benefits More

Without CUDA graphs, each token requires CPU to: run router → pick experts → launch specific GPU kernels → gather results.
This dynamic dispatch per token is the MoE overhead. CUDA graphs record the whole pattern and replay it as one GPU operation.

Dense models have a fixed execution pattern, so the overhead reduction is smaller.

## Known Issues

- **TP=2 under sustained load**: Memory corruption after 8-20 minutes (vLLM issue #35659). Keep `--enforce-eager` for TP=2.
- **Large MoE (122B)**: CUDA graph capture crashes. Too many experts for SM 12.0 graph capture.
- **FlashInfer backend**: Crashes on startup for SM 12.0. Don't use `--attention-backend flashinfer`.
- **Custom all-reduce**: Not supported for SM 12.0 / PCIe. Always use `--disable-custom-all-reduce` for TP=2.

## Configuration

```bash
# Single GPU — CUDA graphs enabled (no enforce-eager)
CUDA_VISIBLE_DEVICES=0 vllm serve model \
    --gpu-memory-utilization 0.90 \
    # NOTE: no --enforce-eager flag

# TP=2 — enforce-eager required for stability
vllm serve model \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --enforce-eager \
```

## References

- vLLM issue #37242: CUDA graphs confirmed working on RTX 5090 (SM 12.0)
- vLLM issue #35659: Memory corruption under sustained TP=2 load
- vLLM issue #30630: Custom all-reduce not supported for SM 12.0
