#!/usr/bin/env python3
"""
Product Quantization (PQ) Vector Search Experiment

Benchmarks PQ compression for the protoResearcher knowledge base using FAISS.
Compares recall@k, search speed, and storage between:
  - Baseline: brute-force float32
  - IVF+PQ: inverted file index with product quantization
  - PQ only: flat product quantization (no IVF)
  - OPQ: optimized product quantization (learned rotation)
  - Binary quantization + rescoring

Uses synthetic vectors matching nomic-embed-text (d=768).

Usage:
  python pq_vector_search.py
  python pq_vector_search.py --n-vectors 10000 --n-queries 100
"""

import argparse
import time

import faiss
import numpy as np
import torch


def generate_realistic_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate unit-normalized clustered vectors mimicking embeddings."""
    rng = np.random.RandomState(seed)
    n_clusters = 20
    cluster_size = n // n_clusters
    vectors = []
    for i in range(n_clusters):
        center = rng.randn(d).astype(np.float32)
        center /= np.linalg.norm(center)
        noise = rng.randn(cluster_size, d).astype(np.float32) * 0.3
        cluster = center[np.newaxis, :] + noise
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        cluster /= norms
        vectors.append(cluster)
    vectors = np.concatenate(vectors, axis=0)[:n]
    return vectors.astype(np.float32)


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    recalls = []
    for i in range(gt.shape[0]):
        gt_set = set(gt[i, :k].tolist())
        pred_set = set(pred[i, :k].tolist())
        recalls.append(len(gt_set & pred_set) / k)
    return np.mean(recalls)


def benchmark_index(name: str, index: faiss.Index, db: np.ndarray,
                    queries: np.ndarray, gt_indices: np.ndarray,
                    train_data: np.ndarray = None):
    """Train, add, search, and report metrics for a FAISS index."""
    d = db.shape[1]
    n = db.shape[0]

    # Train if needed
    t0 = time.time()
    if train_data is not None and hasattr(index, 'train'):
        index.train(train_data)
    train_time = time.time() - t0

    # Add vectors
    t0 = time.time()
    index.add(db)
    add_time = time.time() - t0

    # Search
    t0 = time.time()
    D, I = index.search(queries, 10)
    search_time = time.time() - t0

    # Recall
    r1 = recall_at_k(gt_indices, I, 1)
    r5 = recall_at_k(gt_indices, I, 5)
    r10 = recall_at_k(gt_indices, I, 10)

    # Storage estimate
    if hasattr(index, 'sa_code_size'):
        code_size = index.sa_code_size()
    else:
        code_size = d * 4  # fallback to fp32

    storage_kb = n * code_size / 1024
    fp32_kb = n * d * 4 / 1024
    compression = fp32_kb / storage_kb if storage_kb > 0 else 0

    print(f"\n  {name}")
    print(f"  {'─'*45}")
    print(f"  Train:      {train_time*1000:.1f}ms")
    print(f"  Add:        {add_time*1000:.1f}ms")
    print(f"  Search:     {search_time*1000:.1f}ms ({queries.shape[0]} queries)")
    print(f"  Recall@1:   {r1:.4f} ({r1*100:.1f}%)")
    print(f"  Recall@5:   {r5:.4f} ({r5*100:.1f}%)")
    print(f"  Recall@10:  {r10:.4f} ({r10*100:.1f}%)")
    print(f"  Storage:    {storage_kb:.1f} KB ({compression:.1f}x vs fp32)")

    return {"name": name, "r1": r1, "r5": r5, "r10": r10,
            "storage_kb": storage_kb, "compression": compression,
            "search_ms": search_time * 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=5000)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    d = args.dim
    n = args.n_vectors
    nq = args.n_queries

    print(f"PQ Vector Search Experiment")
    print(f"  Vectors: {n}, Queries: {nq}, Dim: {d}")
    print()

    # Generate data
    db = generate_realistic_vectors(n, d, seed=42)
    queries = generate_realistic_vectors(nq, d, seed=123)

    # Ground truth: brute force
    print("Computing ground truth...")
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(db)
    gt_D, gt_I = index_flat.search(queries, 10)
    print(f"  Done.")

    results = []

    # 1. Flat (baseline)
    print(f"\n{'='*50}")
    print(f"  BASELINES")
    print(f"{'='*50}")
    idx = faiss.IndexFlatIP(d)
    results.append(benchmark_index("Flat FP32 (baseline)", idx, db, queries, gt_I))

    # 2. PQ — various subquantizer counts
    print(f"\n{'='*50}")
    print(f"  PRODUCT QUANTIZATION")
    print(f"{'='*50}")

    for m in [48, 96, 192]:  # subquantizers (d must be divisible by m)
        if d % m != 0:
            continue
        # PQ with m subquantizers, 8 bits each
        idx = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
        results.append(benchmark_index(f"PQ{m}x8 ({m*8//8} bytes/vec)", idx, db, queries, gt_I, db))

    # 3. OPQ — optimized (learned rotation before PQ)
    print(f"\n{'='*50}")
    print(f"  OPTIMIZED PQ (learned rotation)")
    print(f"{'='*50}")

    for m in [48, 96]:
        if d % m != 0:
            continue
        opq = faiss.OPQMatrix(d, m)
        idx_pq = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
        idx = faiss.IndexPreTransform(opq, idx_pq)
        results.append(benchmark_index(f"OPQ{m}x8", idx, db, queries, gt_I, db))

    # 4. IVF+PQ — inverted file for faster search
    print(f"\n{'='*50}")
    print(f"  IVF + PQ (inverted file index)")
    print(f"{'='*50}")

    nlist = min(100, n // 10)  # number of clusters
    for m in [48, 96]:
        if d % m != 0:
            continue
        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = 10  # search 10 clusters
        results.append(benchmark_index(f"IVF{nlist}+PQ{m}x8 (nprobe=10)", idx, db, queries, gt_I, db))

    # 5. Scalar Quantization (SQ8 — 8-bit per dimension)
    print(f"\n{'='*50}")
    print(f"  SCALAR QUANTIZATION")
    print(f"{'='*50}")

    idx = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    results.append(benchmark_index("SQ8 (1 byte/dim)", idx, db, queries, gt_I, db))

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<35} {'R@1':>6} {'R@10':>6} {'Size':>8} {'Comp':>5} {'Search':>8}")
    print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*8} {'─'*5} {'─'*8}")
    for r in results:
        print(f"  {r['name']:<35} {r['r1']:>5.1%} {r['r10']:>5.1%} "
              f"{r['storage_kb']:>6.0f}KB {r['compression']:>4.1f}x {r['search_ms']:>6.1f}ms")

    # Recommendations
    print(f"\n  For protoResearcher (d=768, ~1K-100K vectors):")
    best = max([r for r in results if r['compression'] > 2], key=lambda x: x['r10'])
    print(f"  Recommended: {best['name']}")
    print(f"    Recall@10: {best['r10']:.1%}, {best['compression']:.1f}x compression")


if __name__ == "__main__":
    main()
