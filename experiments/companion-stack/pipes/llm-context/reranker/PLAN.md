# reranker — cross-encoder over embedding retrieval

**Pipe**: llm-context.
**Status**: planned (Phase 2).

## Problem

ORBIS uses Qwen3-Embedding-0.6B for retrieval over its
SQLite-backed memory (`facts`, `sessions` with FTS5,
`personality_events`). Bi-encoder retrieval is fast but
imprecise — cosine similarity over 384-dim vectors misses
fine-grained semantic differences.

## Why ORBIS needs it specifically

The `facts` table is the spine of the companion experience. A
"remember when I said X" query that surfaces 5 *almost-relevant*
facts is a worse experience than 5 *truly-relevant* ones. The
memory curator already runs at half-life decay; pre-filtering with
a reranker before the LLM ever sees results is the cheap quality
lever.

## Architecture

Two-stage retrieval:

1. **Bi-encoder retrieve** (current Qwen3-Embedding-0.6B):
   query → top-50 candidates from FAISS / SQLite vec.
2. **Cross-encoder rerank** (new): for each (query, candidate),
   compute a relevance score; sort and return top-5.

The cross-encoder sees the query and candidate as a single
concatenated input, so it can attend across them — much more
expressive than two independent vectors.

## Candidate models

1. **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (22 M params,
   ~10 ms / pair on GPU) — off-the-shelf strong baseline trained
   on MS MARCO.
2. **`BAAI/bge-reranker-v2-m3`** (568 M params, multilingual) —
   stronger but heavier.
3. **Fine-tune MiniLM on ORBIS-specific (query, fact) pairs** once
   we have data.

For 50 candidates × 10 ms = 500 ms total. Acceptable as an async
pre-LLM step but borderline; consider top-20 input rather than
top-50.

## Datasets

- **MS MARCO** — public, gold-standard reranker training data.
  Off-the-shelf MiniLM is already trained on it.
- **ORBIS memory pairs** — once we have logs, build (query, retrieved-fact, relevant?) tuples by hand or via LLM-as-judge.
- **Self-generated synthetic** — synthesize ORBIS-style queries
  over a held-out facts subset.

## Eval plan

1. **NDCG@5 / NDCG@10** on held-out (query, fact-rank) pairs.
2. **End-to-end LLM answer quality** — A/B test with vs without
   reranker on a 50-question fact-recall set, judge with GPT-4 or
   Qwen-Coder.
3. **Latency profiling** — adds ~500 ms on top of bi-encoder; is
   that worth it for the recall gain?

## Deliverables

- ORBIS integration: a `RerankerService` step in
  ORBIS's memory query path. Probably no novel weights — just a
  thin wrapper around `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Blog post candidate (combined with intent-classifier work):
  "Rebuild your voice agent's brain in three small models."

## Open questions

- Async pre-fetch (rerank during user-still-speaking window) vs
  blocking the LLM call? Probably async if latency matters.
- Top-50 vs top-20 candidates — quality/latency knob.
- Do we need fine-tuning at all, or is off-the-shelf MiniLM good
  enough on ORBIS-style queries?

## Dependencies

- ORBIS fact-recall test set (synthetic for now, real later).
- Existing Qwen3-Embedding service must keep running unchanged.
