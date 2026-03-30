# Context-1 Retrieval Agent Experiment

Evaluating [Chroma Context-1](https://www.trychroma.com/research/context-1) as a retrieval subagent for the protoLabs knowledge system.

## Model

- **chromadb/context-1** — 20B MoE (32 experts, 4 active), Apache 2.0
- Based on OpenAI GPT-OSS-20B, fine-tuned with SFT + RL (CISPO)
- Uses [Harmony message format](https://github.com/openai/harmony) with special tokens
- Trained for agentic multi-hop retrieval with 4 tools

## Benchmark (Single Blackwell GPU, BF16)

| Test | tok/s | Notes |
|------|:-----:|-------|
| Basic generation | 161 | After warmup |
| Tool call generation | 174 | Analysis + tool call |
| Throughput (1024 tok) | **178** | Sustained decode |

## A/B Test: Context-1 + Qwen vs Qwen Solo RAG

### Setup

- **Config A**: Context-1 retrieves documents (iterative multi-hop) → Qwen 35B reasons over retrieved chunks
- **Config B**: Qwen 35B does single-shot keyword search + reasoning directly
- Both configs use the same document store and BM25-lite keyword search

### Results — Small Corpus (10 docs, 20 chunks)

Qwen solo wins. Simple keyword search finds the right docs immediately.

| Metric | Config A | Config B |
|--------|:--:|:--:|
| Avg time | 6.1s | **2.3s** |
| Avg keyword overlap | 0.495 | **0.553** |

### Results — Large Corpus (2,073 docs, 525K chunks)

Context-1 wins decisively on answer quality.

| Metric | Config A | Config B |
|--------|:--:|:--:|
| Avg chunks retrieved | **85** | 10 |
| Avg answer length | **2,986 chars** | 793 chars |
| Avg time | 38.9s | **3.7s** |
| Answer quality | **Detailed, multi-source** | Often "cannot answer" |

Key finding: On a real-size corpus, single-shot keyword search frequently fails to find relevant documents. Context-1's iterative search with query reformulation (3-8 turns) retrieves 8.5x more relevant chunks and produces dramatically better answers. Config B returned "I cannot answer" on several multi-hop queries where Config A found the correct information.

### Quality Examples

**Query**: "How does the protoLabs monitoring stack work end-to-end?"
- **Config A**: 4-section detailed answer covering exporters → scraping → Grafana, citing 90 chunks across repos
- **Config B**: "I cannot answer" — keyword search missed the monitoring docs in 525K chunks

**Query**: "What embedding model does protoResearcher use + what compression experiments were run?"
- **Config A**: Correctly identifies nomic-embed-text, cites multiple sources, describes PQ/SQ8 experiments
- **Config B**: "I cannot answer"

## Latency Gap

Config A is ~10x slower (39s vs 4s). Potential mitigations:
1. Add dense vector search to Config B (embedding similarity, not just keywords)
2. MXFP4 quantization for Context-1 (when checkpoint available)
3. 4x parallel rollouts + RRF fusion (their published best config)
4. Reduce Context-1 turns with better pruning / early stopping

## Architecture

```
experiments/context-1/
├── format.py          # Harmony message format (special tokens, prompt building)
├── tools.py           # 4-tool implementation (search, grep, read, prune) + DocumentStore
├── harness.py         # Agent loop: prompt → generate → parse → execute → repeat
├── ab_test.py         # A/B comparison framework
├── benchmark.py       # Raw throughput benchmark
├── build_corpus.py    # Corpus builder from local repo documentation
├── sample_corpus.json # Small test corpus (10 docs)
├── test_queries.yaml  # Basic test queries
└── hard_queries.yaml  # Multi-hop cross-repo queries
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

# A/B test (requires Context-1 on :8001 + Qwen on :8000)
python -m experiments.context-1.ab_test \
    --corpus experiments/context-1/large_corpus.json \
    --test-queries experiments/context-1/hard_queries.yaml \
    --context1-url http://localhost:8001 \
    --qwen-url http://localhost:8000
```

## vllm-swap configs

```bash
bash models/vllm-swap.sh context-1        # GPU 0, port 8000
bash models/vllm-swap.sh context-1-gpu1   # GPU 1, port 8001 (for A/B testing)
```

## Notes

- Tokenizer patch required: `TokenizersBackend` → `PreTrainedTokenizerFast` in tokenizer_config.json (transformers 5.x feature, vLLM 0.18 uses 4.57)
- Must use `stop_token_ids: [200012, 200002]` (not string stop tokens) for Harmony special tokens
- Must use `skip_special_tokens: false` to preserve Harmony format in completions output
- Must use `--trust-remote-code` for model loading
