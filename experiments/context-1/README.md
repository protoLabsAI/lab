# Context-1 & RAG Technique Evaluation

Comprehensive evaluation of retrieval techniques for the protoLabs knowledge system. Started with [Chroma Context-1](https://www.trychroma.com/research/context-1) as a retrieval subagent, then expanded into a systematic comparison of 10 RAG optimization techniques.

## Final Scorecard

| # | Technique | Quality | Latency | Shipped? |
|---|-----------|:-------:|:-------:|:--------:|
| 1 | **Hybrid search (RRF)** | **+133%** | +50% (5.7s) | **Yes** |
| 2 | **Contextual chunk enrichment** | **+9%** | Zero (index-time) | **Yes** |
| 3 | Cross-encoder reranking | +17% | +58% (8.5s) | Optional |
| 4 | Lost-in-middle reordering | +3% | Zero | Marginal |
| 5 | Multi-query expansion | +1% | +128% (13s) | No |
| 6 | HyDE (hypothetical document) | **-12%** | +74% (9.2s) | **No — hurts** |
| 7 | Contextual compression | **-71%** | -17% (4.4s) | **No — destroys** |
| 8 | Adaptive routing | +11% | +57% (8.3s) | No (latency) |
| 9 | Multi-query + rerank | +0.5% | +170% (15.4s) | No |
| 10 | Rerank + reorder | +18% | +60% (8.8s) | Optional |

**Baseline**: keyword-only search → 856 chars avg answer, 3.8s
**Best default**: hybrid search → 1,994 chars avg answer, 5.7s (+133%)
**Best overall**: hybrid + rerank + reorder → 2,215 chars, 8.8s (+159%)

### What shipped to protoResearcher

1. **Hybrid search** — FTS5 BM25 + sqlite-vec vectors with RRF fusion
2. **Contextual chunk enrichment** — document-level context prepended at index time (Anthropic's technique)
3. Content preview bumped 200 → 1000 chars
4. Middleware top_k bumped 5 → 10

## Context-1 Model

- **chromadb/context-1** — 20B MoE (32 experts, 4 active), Apache 2.0
- Based on OpenAI GPT-OSS-20B, [Harmony message format](https://github.com/openai/harmony)
- Trained for agentic multi-hop retrieval (SFT + RL/CISPO)
- **178 tok/s** on single Blackwell GPU (BF16)

## Context-1 A/B Test

### Small Corpus (10 docs, 20 chunks)

Qwen solo wins — keyword search finds the right docs immediately.

| Metric | Context-1 + Qwen | Qwen Solo |
|--------|:--:|:--:|
| Avg time | 6.1s | **2.3s** |
| Keyword overlap | 0.495 | **0.553** |

### Large Corpus (2,073 docs, 525K chunks)

Context-1 wins decisively on answer quality.

| Metric | Context-1 + Qwen | Qwen Keyword | Qwen Hybrid |
|--------|:--:|:--:|:--:|
| Chunks retrieved | **85** | 10 | 20 |
| Answer length | **2,986** | 793 | **2,149** |
| Avg time | 42.9s | 3.7s | **5.9s** |

Key finding: Hybrid search closes 77% of Context-1's quality gap at 7x less latency. Context-1's multi-hop iterative search is overkill when you have good vector+keyword fusion.

## RAG Technique Deep Dives

### 1. Hybrid Search (RRF) — **+133%, shipped**

Reciprocal Rank Fusion of keyword (BM25 via FTS5) and dense vector (FAISS + MiniLM) results. The single biggest improvement. Keyword search alone frequently returned "cannot answer" on 525K chunks — dense vectors find semantically similar docs that keywords miss, and vice versa.

### 2. Contextual Chunk Enrichment — **+9%, shipped**

Based on [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). Prepends document-level context to chunks before embedding. Only 2/5 top-5 chunks overlap with non-enriched baseline — fundamentally changes what gets retrieved. Biggest wins on cross-repo queries (secrets management +91%, monitoring stack +40%).

### 3. Cross-Encoder Reranking — **+17%, optional**

mxbai-rerank-xsmall-v1 (30M params, CPU). Retrieves 40 candidates via hybrid search, reranks with cross-encoder, returns top 20. Helps on nuanced queries but adds ~2.7s CPU latency.

### 4. Lost-in-Middle Reordering — **+3%, free**

Reorders retrieved chunks so most relevant are at positions 1 and N (start/end of context), less relevant in middle. Combats LLM U-shaped attention bias. Minimal effect at 20 chunks / ~10K tokens — more impactful at longer contexts.

### 5. Multi-Query Expansion — **+1%, not worth it**

LLM generates 3 query variants, searches each, RRF fuses all results. The LLM call takes ~7s and barely improves recall over hybrid search alone.

### 6. HyDE — **-12%, harmful**

Generates a hypothetical answer, embeds it, searches for similar docs. On technical corpora, the hypothetical introduces incorrect vocabulary that pulls in irrelevant chunks. Works better for generic web content.

### 7. Contextual Compression — **-71%, destructive**

LLM extracts "relevant sentences" from each chunk before feeding to the reasoning model. Over-extracts and destroys context — the reasoning model needs surrounding sentences to understand the chunk.

### 8. Adaptive Routing — **+11%, too slow**

Classifies query complexity (simple/moderate/complex), routes to different search strategies. Complex queries get reranking, simple queries get small-k vector only. The classifier LLM call adds ~2.5s, eating the latency savings.

## Architecture

```
experiments/context-1/
├── format.py                  # Harmony message format (special tokens, prompt building)
├── tools.py                   # DocumentStore with keyword/dense/hybrid/rerank search
├── harness.py                 # Context-1 agent loop (observe → reason → act)
├── ab_test.py                 # A/B comparison framework (Config A vs Config B)
├── benchmark.py               # Raw throughput benchmark
├── compare_search.py          # Multi-mode search comparison (all 10 techniques)
├── contextual_enrichment.py   # Anthropic contextual retrieval experiment
├── build_corpus.py            # Corpus builder from local repo documentation
├── sample_corpus.json         # Small test corpus (10 docs)
├── large_corpus.json          # Full corpus (2K+ docs, gitignored)
├── large_corpus.faiss         # Pre-built FAISS index (gitignored)
├── test_queries.yaml          # Basic test queries
├── hard_queries.yaml          # Multi-hop cross-repo queries
├── ab_results.json            # Small corpus A/B results
├── ab_results_large.json      # Large corpus A/B results (keyword Config B)
├── ab_results_hybrid.json     # Large corpus A/B results (hybrid Config B)
├── compare_results.json       # Keyword vs hybrid vs rerank
├── compare_results_v2.json    # + multi-query expansion
├── compare_results_v4.json    # + HyDE, compression, adaptive
└── contextual_results.json    # Contextual enrichment A/B
```

## Running

```bash
# Serve Context-1 (single GPU)
bash models/vllm-swap.sh context-1

# Benchmark throughput
python -m experiments.context-1.benchmark

# Run agent on a query
python -m experiments.context-1.harness \
    --query "How does FP8 quantization work?" \
    --corpus experiments/context-1/sample_corpus.json

# Build large corpus from local repos
python -m experiments.context-1.build_corpus

# Compare search techniques (requires pre-built FAISS index)
python experiments/context-1/compare_search.py \
    --corpus experiments/context-1/large_corpus.json \
    --queries experiments/context-1/hard_queries.yaml \
    --index-path experiments/context-1/large_corpus.faiss

# Test contextual enrichment
python experiments/context-1/contextual_enrichment.py \
    --corpus experiments/context-1/large_corpus.json \
    --queries experiments/context-1/hard_queries.yaml

# A/B test (requires Context-1 on :8001 + Qwen on :8000)
python -m experiments.context-1.ab_test \
    --corpus experiments/context-1/large_corpus.json \
    --test-queries experiments/context-1/hard_queries.yaml \
    --context1-url http://localhost:8001 \
    --qwen-url http://localhost:8000 \
    --search-mode hybrid
```

## vllm-swap configs

```bash
bash models/vllm-swap.sh context-1        # GPU 0, port 8000
bash models/vllm-swap.sh context-1-gpu1   # GPU 1, port 8001 (for A/B testing)
```

## Technical Notes

- Tokenizer patch required: `TokenizersBackend` → `PreTrainedTokenizerFast` in tokenizer_config.json (transformers 5.x feature, vLLM 0.18 uses 4.57)
- Must use `stop_token_ids: [200012, 200002]` (not string stop tokens) for Harmony special tokens
- Must use `skip_special_tokens: false` to preserve Harmony format in completions output
- Must use `--trust-remote-code` for model loading
- FAISS index: 525K vectors (all-MiniLM-L6-v2, 384-dim) embedded in 135s on GPU, <5ms search
- Reranker: mxbai-rerank-xsmall-v1, 30M params, ~70ms for 20 pairs on CPU

## Key Takeaway

The two biggest wins are fundamentally different in nature:
- **Hybrid search** fixes *how you find* documents (retrieval algorithm)
- **Contextual enrichment** fixes *what you index* (data quality)

Everything else optimizes at the margins. On a technical corpus at sub-million scale, a good foundation (hybrid search + enriched chunks + capable LLM) outperforms complex retrieval architectures like dedicated retrieval agents or multi-stage pipelines.
