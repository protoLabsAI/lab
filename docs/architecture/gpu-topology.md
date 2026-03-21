# GPU Topology & Model Deployment

## Hardware

- 2x NVIDIA RTX PRO 6000 Blackwell
- 96 GB GDDR7 VRAM each (192 GB total)
- PCIe Gen5 x16 (NOT NVLink)
- SM 12.0 compute capability
- TDP 600W each (draw ~300-340W during inference)

## Deployment Modes

### Single GPU (GPU 0)

Best for: daily driver models, experiments on GPU 1 free for ComfyUI/video gen.

| Model | Weights | KV Room | Max Context | tok/s |
|-------|---------|---------|-------------|-------|
| Qwen 27B INT4 | 14GB | 68GB | 160K | 44 |
| Qwen 35B MoE BF16 | 72GB | 14GB | 64K | 170 |
| Qwen 122B INT4 | 35GB | 51GB | 64K | ~30 |
| OmniCoder 9B | 18GB | 68GB | 262K | 92 |
| Llama 70B AWQ | 40GB | 46GB | 128K | 38 |

### Dual GPU (TP=2)

Best for: maximum context or quality. Uses both GPUs — no ComfyUI.

| Model | Weights/GPU | KV Room/GPU | Max Context | Notes |
|-------|-------------|-------------|-------------|-------|
| Qwen 35B MoE BF16 | 36GB | 50GB | 250K | Best config: 3/4 pass^3, 170 tok/s |
| Qwen 122B FP8 | 35GB | 51GB | 64-128K | High quality, enforce-eager only |
| Qwen 122B INT4 | 17GB | 69GB | 128K | enforce-eager only |
| Qwen 27B INT4 | 7GB | 79GB | 256K+ | Massive concurrency headroom |

## CUDA Graph Compatibility

| Model Type | Single GPU | TP=2 |
|-----------|:---:|:---:|
| Dense (27B, 70B) | CUDA graphs (1.3-1.5x) | enforce-eager (memory corruption) |
| Small MoE (35B, 3B active) | CUDA graphs (**5.7x!!**) | No enforce-eager needed |
| Large MoE (122B, 10B active) | **CRASHES** | enforce-eager required |

## Power Budget

| Scenario | GPU Draw | System Total | UPS Requirement |
|----------|----------|-------------|-----------------|
| Single GPU inference | 300-340W | ~600W | 1000W OK |
| TP=2 inference | 600-680W | ~900W | 1000W tight |
| TP=2 + model load spike | 800-1000W | ~1200W | 1600W needed |
| Training (LoRA) | 400-500W | ~750W | 1000W OK |
