#!/usr/bin/env python3
"""Evaluate reranker models on the synthetic fact-recall benchmark.

Pipeline:
1. Embed all facts via Qwen3-Embedding-0.6B (:8001)
2. For each query: retrieve top-K via cosine similarity
3. Rerank with each candidate model
4. Measure NDCG@5, MRR, latency

Baselines: random order, bi-encoder only (no rerank), BM25 only.
Candidates: Qwen3-Reranker-0.6B (via :8001), mxbai-rerank-xsmall-v1 (local).

Usage:
    python scripts/eval_rerankers.py
    python scripts/eval_rerankers.py --top-k 20      # fewer candidates
    python scripts/eval_rerankers.py --embed-url http://protolabs:8001
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import requests


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def embed_texts(texts: list[str], base_url: str) -> np.ndarray:
    """Embed texts via the Qwen3-Embedding service."""
    # Batch in chunks of 32
    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        resp = requests.post(
            f"{base_url}/v1/embeddings",
            json={"input": batch, "model": "qwen3-embedding"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        embeddings = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all docs."""
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def rerank_via_api(
    query: str, documents: list[str], base_url: str
) -> list[float]:
    """Rerank via the embed server's /v1/rerank endpoint (Qwen3-Reranker)."""
    resp = requests.post(
        f"{base_url}/v1/rerank",
        json={
            "query": query,
            "documents": documents,
            "model": "qwen3-reranker",
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    scores = [0.0] * len(documents)
    for r in results:
        scores[r["index"]] = r["relevance_score"]
    return scores


def rerank_via_cross_encoder(
    query: str, documents: list[str], model
) -> list[float]:
    """Rerank via a local sentence-transformers CrossEncoder."""
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    return scores.tolist()


def ndcg_at_k(ranked_ids: list[int], gold_ids: set[int], k: int) -> float:
    """Compute NDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(i + 2)
    # Ideal DCG
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr(ranked_ids: list[int], gold_ids: set[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(
    queries: list[dict],
    ranker_fn,
    name: str,
    k: int = 5,
) -> dict:
    """Evaluate a ranking function on all queries."""
    ndcgs = []
    mrrs = []
    latencies = []

    for q in queries:
        gold_ids = set(q["gold_fact_ids"])
        candidates = q["candidate_ids"]
        candidate_texts = q["candidate_texts"]

        t0 = time.time()
        scores = ranker_fn(q["query"], candidate_texts, candidates)
        latency = time.time() - t0
        latencies.append(latency)

        # Sort by score descending
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        ranked_ids = [r[0] for r in ranked]

        ndcgs.append(ndcg_at_k(ranked_ids, gold_ids, k))
        mrrs.append(mrr(ranked_ids, gold_ids))

    results = {
        "name": name,
        "ndcg@5": float(np.mean(ndcgs)),
        "mrr": float(np.mean(mrrs)),
        "latency_mean_ms": float(np.mean(latencies) * 1000),
        "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
    }
    print(f"\n=== {name} ===")
    print(f"  NDCG@5: {results['ndcg@5']:.4f}")
    print(f"  MRR:    {results['mrr']:.4f}")
    print(f"  Latency: {results['latency_mean_ms']:.1f}ms mean, {results['latency_p95_ms']:.1f}ms p95")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate reranker models")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--embed-url", type=str, default="http://localhost:8001")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K candidates from bi-encoder")
    parser.add_argument("--eval-k", type=int, default=5, help="NDCG@K evaluation depth")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent / "data")
    data_dir = Path(args.data_dir)

    facts = load_jsonl(data_dir / "facts.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    print(f"Facts: {len(facts)}, Queries: {len(queries)}")

    # Check embed service
    try:
        r = requests.get(f"{args.embed_url}/health", timeout=5)
        print(f"Embed service: {args.embed_url} (status: {r.status_code})")
    except requests.ConnectionError:
        print(f"ERROR: Embed service not reachable at {args.embed_url}")
        return

    # Embed all facts
    print("\nEmbedding facts...")
    fact_texts = [f["text"] for f in facts]
    t0 = time.time()
    fact_embeddings = embed_texts(fact_texts, args.embed_url)
    print(f"  {len(fact_texts)} facts embedded in {time.time()-t0:.1f}s, shape={fact_embeddings.shape}")

    # Embed all queries
    print("Embedding queries...")
    query_texts = [q["query"] for q in queries]
    t0 = time.time()
    query_embeddings = embed_texts(query_texts, args.embed_url)
    print(f"  {len(query_texts)} queries embedded in {time.time()-t0:.1f}s")

    # Retrieve top-K candidates for each query
    print(f"\nRetrieving top-{args.top_k} candidates per query...")
    for i, q in enumerate(queries):
        sims = cosine_similarity(query_embeddings[i], fact_embeddings)
        top_ids = np.argsort(sims)[::-1][: args.top_k].tolist()
        q["candidate_ids"] = top_ids
        q["candidate_texts"] = [fact_texts[j] for j in top_ids]
        q["candidate_scores"] = [float(sims[j]) for j in top_ids]

    # Baseline 1: Random ranking
    results = []
    results.append(evaluate_ranking(
        queries,
        lambda query, texts, ids: [random.random() for _ in ids],
        "random",
        k=args.eval_k,
    ))

    # Baseline 2: Bi-encoder only (cosine similarity order)
    results.append(evaluate_ranking(
        queries,
        lambda query, texts, ids: [
            queries[0]["candidate_scores"][queries[0]["candidate_ids"].index(i)]
            if i in queries[0]["candidate_ids"]
            else 0.0
            for i in ids
        ],
        "bi-encoder-only",
        k=args.eval_k,
    ))

    # Fix: bi-encoder ranking should use stored scores properly
    def bi_encoder_ranker(query, texts, ids):
        # Find this query in the queries list and return pre-computed scores
        for q in queries:
            if q["query"] == query:
                return [q["candidate_scores"][q["candidate_ids"].index(i)] for i in ids]
        return [0.0] * len(ids)

    results[-1] = evaluate_ranking(queries, bi_encoder_ranker, "bi-encoder-only", k=args.eval_k)

    # Candidate 1: Qwen3-Reranker via :8001 /v1/rerank
    try:
        results.append(evaluate_ranking(
            queries,
            lambda query, texts, ids: rerank_via_api(query, texts, args.embed_url),
            "qwen3-reranker-0.6b",
            k=args.eval_k,
        ))
    except Exception as e:
        print(f"\nQwen3-Reranker FAILED: {e}")

    # Candidate 2: mxbai-rerank-xsmall-v1 (local cross-encoder)
    try:
        from sentence_transformers import CrossEncoder
        mxbai = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
        results.append(evaluate_ranking(
            queries,
            lambda query, texts, ids: rerank_via_cross_encoder(query, texts, mxbai),
            "mxbai-rerank-xsmall-v1",
            k=args.eval_k,
        ))
    except Exception as e:
        print(f"\nmxbai-rerank FAILED: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'NDCG@5':>8} {'MRR':>8} {'Latency':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['ndcg@5']:>8.4f} {r['mrr']:>8.4f} {r['latency_mean_ms']:>8.1f}ms")
    print("=" * 70)

    # Save results
    results_path = data_dir.parent / "eval" / "reranker_results.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
