# Delta Review — 2026-04-26

Changes from original PLAN.md based on current infrastructure.

## ORBIS has NO vector retrieval today

PLAN assumes Qwen3-Embedding retrieval is wired into ORBIS. It is
not. ORBIS memory is SQLite FTS5 BM25 only (`facts_fts`,
`sessions_fts`). The embed service on :8001 exists but nothing in
ORBIS calls it.

**Implication:** the reranker experiment scope expands to include
wiring bi-encoder retrieval into ORBIS as a prerequisite, OR we
eval the reranker standalone on a synthetic fact-recall dataset and
defer ORBIS integration.

Decision: **eval standalone first.** Build a fact-recall benchmark,
measure off-the-shelf reranker quality, then wire into ORBIS as
an engineering PR.

## Available reranker models (already on disk)

| Model | Size | Where |
|---|---|---|
| `Qwen3-Reranker-0.6B` | 1.2 GB | `/mnt/models/huggingface/hub/` — also served via `:8001` `/v1/rerank` |
| `mxbai-rerank-xsmall-v1` | 146 MB | `/mnt/models/huggingface/hub/` — benchmarked in context-1 (30M params, ~70ms/20 pairs) |
| `all-MiniLM-L6-v2` | 88 MB | Downloaded, 384-dim bi-encoder (not a reranker) |

`cross-encoder/ms-marco-MiniLM-L-6-v2` from the original PLAN is
NOT downloaded. We have better options already on disk.

## Embedding dimensions

PLAN says "384-dim vectors" — wrong. Qwen3-Embedding-0.6B outputs
**2560-dim** (benchmarked at 331.5 docs/s, 17ms single query).

## Eval approach (standalone, no ORBIS wiring yet)

1. Generate synthetic ORBIS-style fact-recall dataset: 100 facts +
   100 queries, each query has 1-3 gold facts.
2. Retrieve top-20 via Qwen3-Embedding (`:8001`).
3. Rerank with each candidate model.
4. Measure NDCG@5, MRR, latency.
5. Baselines: BM25-only (FTS5), bi-encoder-only (no rerank),
   random rerank.

## Latency budget

context-1 benchmarked mxbai-rerank at ~70ms for 20 pairs on CPU.
For ORBIS voice loop, 100ms total rerank budget means top-20 input
is fine. top-50 at 10ms/pair GPU is also acceptable.
