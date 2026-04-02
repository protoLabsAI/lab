# RAG Embedding Benchmark

Comparing embedding models and serving approaches for protoResearcher's RAG pipeline.

## Results (2026-04-02)

### Embedding Latency

| Config | Dims | 1-query | Batch(5) | Batch(20) | docs/s | VRAM |
|--------|:----:|:-------:|:--------:|:---------:|:------:|:----:|
| **Ollama 0.6B** (ava node) | 1024 | 106ms | 279ms | 785ms | 25.5 | Remote |
| **Local 0.6B** (Blackwell GPU 1) | 1024 | **13ms** | **22ms** | **36ms** | **562** | 1.2 GB |
| **Local 4B** (Blackwell GPU 1) | 2560 | 17ms | 36ms | 60ms | 332 | 8.5 GB |

**Key finding**: Local GPU serving is **8x faster single-query** and **22x higher throughput** than Ollama over Tailscale. The 0.6B model on Blackwell GPU achieves 562 docs/s — enough to re-embed the entire 525K chunk corpus in ~15 minutes.

### Embedding Quality (Cosine Similarity Discrimination)

| Config | Avg Similar | Avg Dissimilar | Discrimination |
|--------|:-----------:|:--------------:|:--------------:|
| **Ollama 0.6B** | 0.7267 | 0.3622 | **0.3645** |
| **Local 0.6B** | 0.7113 | 0.3488 | 0.3625 |
| **Local 4B** | 0.7797 | 0.4276 | 0.3521 |

Quality is nearly identical across all configs — same model weights produce same embeddings regardless of serving infrastructure. The 4B model has higher absolute similarity scores but slightly lower discrimination (it scores dissimilar pairs higher too). For RAG retrieval, **discrimination matters more than absolute scores** — 0.6B wins.

### Recommendation

**Switch to local 0.6B** on GPU 1. Same quality, 8-22x faster, eliminates Ollama dependency and Tailscale network hop. Only 1.2GB VRAM — negligible on a 96GB GPU.

The 4B model doesn't justify its 7x VRAM cost for marginal quality difference. Save that VRAM for the LLM or reranker.

## Architecture

```
                    ┌─────────────────────┐
                    │   protoResearcher   │
                    │   (port 7872)       │
                    └──────┬──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌──────────┐ ┌──────────┐
         │ BM25    │ │ Embed    │ │ Reranker │
         │ FTS5    │ │ 0.6B     │ │ 0.6B     │
         │ sqlite  │ │ GPU 1    │ │ GPU 1    │
         └─────────┘ │ :8001    │ │ :8002    │
                     └──────────┘ └──────────┘
                           │
                    ┌──────┴──────────────┐
                    │   Hybrid RRF        │
                    │   (BM25 + Vector)   │
                    └─────────────────────┘
```

## Serving

```bash
# Start embedding server on GPU 1 (1.2GB VRAM)
CUDA_VISIBLE_DEVICES=1 uv run python experiments/rag-bench/serve_embed.py \
  --model Qwen/Qwen3-Embedding-0.6B --port 8001

# Or 4B for higher quality (8.5GB VRAM)
CUDA_VISIBLE_DEVICES=1 uv run python experiments/rag-bench/serve_embed.py \
  --model Qwen/Qwen3-Embedding-4B --port 8001

# API is OpenAI-compatible
curl http://localhost:8001/v1/embeddings -d '{"input":"test query"}'
```

## Why Not Shared Encoder?

Qwen LLM and Qwen Embedding share the same transformer architecture, but:
- **Embeddings need bidirectional attention** (see full sequence) while generation needs causal attention (see only left context)
- vLLM can't serve both `generate` and `embed` from one instance
- GritLM solves this with instruction-switching but only exists for Llama
- The 0.6B embedding model is so small (1.2GB) that a separate instance is the pragmatic choice

## Sorted Chunk Optimization

vLLM's prefix caching reuses KV cache for repeated prefixes. For RAG:
- Sort retrieved chunks by a deterministic key (chunk ID) in the prompt
- Queries retrieving overlapping documents share prefix cache
- Measured: TTFT drops from 3077ms → 29ms with prefix caching on 122B

## Files

- `serve_embed.py` — Lightweight FastAPI embedding server (transformers-based)
- `bench.py` — Embedding latency + quality benchmark
- `results/` — Benchmark outputs
