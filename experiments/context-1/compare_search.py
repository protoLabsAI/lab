"""Compare search modes: keyword vs hybrid vs hybrid+rerank.

Runs Config B (Qwen solo RAG) across all search modes on the same queries
to measure the impact of each retrieval improvement.

Usage:
    python -m experiments.context-1.compare_search \
        --corpus experiments/context-1/large_corpus.json \
        --queries experiments/context-1/hard_queries.yaml \
        --index-path experiments/context-1/large_corpus.faiss
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tools import DocumentStore


QWEN_RAG_SYSTEM = """\
You are a research assistant. Answer the user's question using ONLY the provided context documents. \
If the context doesn't contain enough information, say so. Cite specific documents when possible."""

QUERY_EXPAND_SYSTEM = """\
Generate 3 alternative search queries for the given question. Each query should \
approach the topic from a different angle or use different keywords. Return ONLY \
the queries, one per line, no numbering or explanation."""


def qwen_chat(messages, url, model="local", max_tokens=2048):
    resp = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def expand_query(query: str, url: str, model: str = "local") -> list[str]:
    """Generate query variants using LLM. Returns original + 3 expansions."""
    try:
        result = qwen_chat(
            [
                {"role": "system", "content": QUERY_EXPAND_SYSTEM},
                {"role": "user", "content": query},
            ],
            url, model=model, max_tokens=200,
        )
        variants = [line.strip().lstrip("0123456789.-) ") for line in result.strip().split("\n") if line.strip()]
        return [query] + variants[:3]
    except Exception:
        return [query]


def multi_query_search(query, store, qwen_url, k=20, use_rerank=False):
    """Expand query into variants, search each, RRF fuse all results."""
    queries = expand_query(query, qwen_url)

    # Collect results from all query variants
    all_chunks: dict[str, tuple[int, any]] = {}  # chunk_id -> (best_rank, chunk)
    rrf_scores: dict[str, float] = {}
    rrf_k = 60

    for qi, q in enumerate(queries):
        results = store.hybrid_search(q, k=k)
        for rank, chunk in enumerate(results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            if chunk.chunk_id not in all_chunks:
                all_chunks[chunk.chunk_id] = chunk

    # Sort by RRF score
    ranked_ids = sorted(rrf_scores.items(), key=lambda x: -x[1])
    fused = [all_chunks[cid] for cid, _ in ranked_ids[:k * 2 if use_rerank else k]]

    if use_rerank:
        return store.rerank(query, fused, top_k=k)
    return fused[:k]


def run_query(query, store, qwen_url, search_mode, k=20):
    t0 = time.time()

    # Retrieve
    t_search = time.time()
    if search_mode == "multi-query":
        chunks = multi_query_search(query, store, qwen_url, k=k, use_rerank=False)
    elif search_mode == "multi-query+rerank":
        chunks = multi_query_search(query, store, qwen_url, k=k, use_rerank=True)
    elif search_mode == "hybrid+rerank":
        chunks = store.hybrid_rerank_search(query, k=k, candidates=k * 2)
    elif search_mode == "hybrid":
        chunks = store.hybrid_search(query, k=k)
    elif search_mode == "dense":
        chunks = store.dense_search(query, k=k)
    else:
        chunks = store.search(query, k=k)
    search_ms = (time.time() - t_search) * 1000

    # Reason
    context_text = "\n\n---\n\n".join(
        f"[{c.chunk_id}]\n{c.content}" for c in chunks
    )
    messages = [
        {"role": "system", "content": QWEN_RAG_SYSTEM},
        {"role": "user", "content": f"Context documents:\n\n{context_text}\n\n---\n\nQuestion: {query}"},
    ]
    answer = qwen_chat(messages, qwen_url)
    elapsed = time.time() - t0

    return {
        "answer": answer,
        "answer_length": len(answer),
        "chunks": len(chunks),
        "search_ms": round(search_ms),
        "total_s": round(elapsed, 2),
        "chunk_ids": [c.chunk_id for c in chunks[:5]],  # top 5 for comparison
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qwen-url", default="http://localhost:8000")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--output", default="experiments/context-1/compare_results.json")
    args = parser.parse_args()

    store = DocumentStore()
    n = store.load_from_json(Path(args.corpus))
    print(f"Loaded {n} items, {len(store.chunks)} chunks")

    # Load dense index
    if args.index_path and Path(args.index_path).exists():
        store.load_dense_index(Path(args.index_path), device="cpu")
    else:
        store.build_dense_index(device="cpu")

    # Load reranker
    store.load_reranker()

    queries = yaml.safe_load(Path(args.queries).read_text())
    modes = ["keyword", "hybrid", "hybrid+rerank", "multi-query", "multi-query+rerank"]

    results = []
    for i, q in enumerate(queries):
        query = q["query"]
        print(f"\n{'='*60}")
        print(f"Query {i+1}/{len(queries)}: {query[:60]}...")

        row = {"query": query}
        for mode in modes:
            r = run_query(query, store, args.qwen_url, mode)
            row[mode] = r
            print(f"  {mode:>15}: {r['search_ms']:>4}ms search, {r['total_s']:>5.1f}s total, {r['answer_length']:>5} chars")

        results.append(row)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Query':<45} {'keyword':>8} {'hybrid':>8} {'rerank':>8}")
    print(f"{'':45} {'len/time':>8} {'len/time':>8} {'len/time':>8}")
    print("-" * 75)
    for r in results:
        q = r["query"][:42] + "..." if len(r["query"]) > 42 else r["query"]
        kw = r["keyword"]
        hy = r["hybrid"]
        rr = r["hybrid+rerank"]
        print(f"{q:<45} {kw['answer_length']:>4}/{kw['total_s']:.1f}s {hy['answer_length']:>4}/{hy['total_s']:.1f}s {rr['answer_length']:>4}/{rr['total_s']:.1f}s")

    # Averages
    n = len(results)
    for mode in modes:
        avg_len = sum(r[mode]["answer_length"] for r in results) / n
        avg_time = sum(r[mode]["total_s"] for r in results) / n
        avg_search = sum(r[mode]["search_ms"] for r in results) / n
        print(f"\n{mode}: avg {avg_len:.0f} chars, {avg_time:.1f}s total, {avg_search:.0f}ms search")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
