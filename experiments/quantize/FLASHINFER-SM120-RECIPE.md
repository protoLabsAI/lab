# FlashInfer on Blackwell sm120 — working build recipe (2026-06-13)

**Result:** flashinfer 0.6.11.post2 cutlass fused-MoE kernel **compiles + loads on sm120**
(RTX PRO 6000 Blackwell, driver 595, vLLM 0.22.1, CUDA 13). This breaks the long-standing
"FlashInfer requires sm75 or higher" / "No supported CUDA architectures for [12]" wall that
blocked: bf16 MoE TP=2 (35B), FP8-KV cache (`--kv-cache-dtype fp8`, 2× context), nano-LFM2.5,
Mistral-Medium-128B.

## Root cause chain (5 layers, each masked the next)

1. **Arch detect** — `flashinfer/compilation_context.py` auto-detects via `get_device_capability`,
   but `_normalize_cuda_arch` raises `SM 12.x requires CUDA >= 12.9` because **system nvcc 12.8 is
   first in PATH** → empty arch set → `No supported CUDA architectures for [12]`.
2. **nvcc version** — system 12.8 nvcc rejects `compute_120f` (`nvcc fatal: Unsupported gpu architecture`).
3. **cccl guard** — pip wheels are skewed: `nvidia-cuda-nvcc 13.3.33` vs cudart headers **13.0**
   (`CUDA_VERSION 13000`); cccl `cuda_toolkit.h` strict major.minor equality check errors.
4. **build OOM** — ninja at full parallelism + prod models resident → OOM-killer kills nvcc
   (`Killed`, no `error:` line). Reached 96/97 before dying.
5. **link** — `ld` searches `cu13/lib64` (doesn't exist; libs are in `cu13/lib`) and the wheels ship
   versioned `libcudart.so.13` without bare `.so` → `cannot find -lcudart/-lnvrtc`.

## The fix (env + symlinks)

```bash
CU13=/home/ava/dev/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13
export PATH=$CU13/bin:$PATH                 # CUDA-13.3 nvcc, not system 12.8
export FLASHINFER_CUDA_ARCH_LIST="12.0f"    # emit compute_120f, skip broken auto-detect
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"   # 13.3-nvcc vs 13.0-cudart skew
export MAX_JOBS=4                            # avoid OOM during cutlass compile (RAM, not VRAM)
# one-time linker bridge for the fragmented pip CUDA layout:
ln -sf lib  $CU13/lib64
ln -sf libcudart.so.13 $CU13/lib/libcudart.so
ln -sf libnvrtc.so.13  $CU13/lib/libnvrtc.so
```

Verified: `gen_cutlass_fused_moe_sm120_module(False).build_and_load()` → OK.
JIT cache: `~/.cache/flashinfer/0.6.11.post2/120f/fused_moe_120.so` (subsequent loads ~2s).

## Status
- [x] **Wired into `models/vllm-swap.sh`** (export block, guarded on cu13 dir present).
- [x] **End-to-end validated 2026-06-13:** `qwen-36b-bf16` (35B-A3B bf16 hybrid-Mamba MoE) serves via
  "FlashInfer CUTLASS Unquantized MoE" on sm120 at **173.9 tok/s** (single GPU, util 0.85). Also needed
  `--max-num-seqs 512` (0.22.1: default 1024 > 566 Mamba cache blocks → blocks cudagraph capture).
- [ ] Bigger prizes behind the same wall: **FP8-KV cache** (`--kv-cache-dtype fp8`, flashinfer attention
  path — separate kernel, not yet tested), nano-LFM2, Mistral-128B.
- Cleaner long-term fix: align the `nvidia-cuda-*` wheel versions (cudart → 13.3) so the cccl guard
  passes natively and `lib64`/`.so` symlinks aren't needed.
