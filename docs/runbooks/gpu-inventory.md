# GPU Inventory — protolabs (Blackwell node)

> Last updated: 2026-04-27

## Running Services

| GPU | Port | Service | Model | VRAM | Ctx | Mode | Notes |
|-----|------|---------|-------|------|-----|------|-------|
| **0** | :8000 | vLLM (`local`) | Qwen3.6-27B-FP8 | ~91 GB | 256K | Thinking, agentic, vision | Official Qwen FP8. Primary for planning, tool use, image understanding |
| **1** | :8001 | Embed server | Qwen3-Embedding-0.6B | ~2 GB | — | Embeddings | 562 docs/s, always-on |
| **1** | :8002 | vLLM (`local-voice`) | Qwen3.6-35B-A3B-uncensored-heretic (on-the-fly FP8) | ~35 GB | 128K | No-thinking, text-only | Uncensored finetune by llmfan46. **Vision broken** — degenerate `!!!!` output on images, text-only finetune corrupted VL alignment. `--language-model-only` required. Eval: 10/10 suites, comm 0.75, creative 24/25, FC 8/8 |
| **1** | :8092 | Fish Audio S2-Pro | fishaudio/s2-pro | ~20 GB | — | TTS | 80+ languages, voice cloning |
| **1** | :8188 | ComfyUI | — | ~0.5 GB | — | Image/video gen | |
| — | :9100 | node-exporter | — | — | — | Prometheus metrics | Docker |
| — | :8080 | cadvisor | — | — | — | Docker metrics | Docker |
| — | :9835 | nvidia-gpu-exporter | — | — | — | GPU metrics | Docker |

## GPU Memory

- **GPU 0:** ~91/98 GB (93%) — 27B owns it
- **GPU 1:** ~58/98 GB (~59%) — 35B + embed + TTS + ComfyUI, headroom for video gen

## Gateway Routing (LiteLLM on ava:4000)

- `protolabs/smart` → :8000 (27B, thinking)
- `protolabs/fast` → :8002 (35B uncensored, no-thinking)

## Key Constraints

- **Vision only through 27B (`local`)** — 35B uncensored finetune has broken VL alignment (degenerate `!!!!` on image inputs)
- 35B served with `--quantization fp8` (on-the-fly, no separate quant files needed)
- `CUDA_VISIBLE_DEVICES` doesn't propagate via bash subshells to vLLM's multiprocessing workers — must use systemd `Environment=` for GPU pinning
- `--attention-backend TRITON_ATTN` required on Blackwell (FlashInfer crashes on sm120)
- `vllm-swap.sh dual` config available to start both models via systemd

## Systemd Services

```bash
# Start both models
bash models/vllm-swap.sh dual

# Individual control
sudo systemctl {start|stop|restart|status} vllm        # 27B on GPU 0
sudo systemctl {start|stop|restart|status} vllm-voice   # 35B on GPU 1
sudo systemctl {start|stop|restart|status} embed-server  # Embeddings on GPU 1
```

## Eval Results — Uncensored 35B (quick profile, no-thinking, 128K ctx)

| Suite | Result |
|-------|--------|
| claw-eval | ✅ PASS |
| coding | ✅ PASS |
| instruction_following | ✅ PASS |
| reasoning | ✅ PASS |
| structured_output | ✅ PASS |
| summarization | ✅ PASS |
| safety | ✅ PASS |
| creative_writing | ✅ PASS (24/25) |
| roleplay | ✅ PASS (5/5) |
| function_call | ✅ PASS (8/8, 100%) |
