#!/usr/bin/env python3
"""
TurboQuant Vector Search Experiment

Tests TurboQuant compression on the protoResearcher knowledge base vectors.
Compares recall@k, search speed, and storage between:
  - Baseline: float32 brute-force (sqlite-vec style)
  - TurboQuant 3-bit: ~10x compression
  - TurboQuant 4-bit: ~8x compression

Uses synthetic vectors matching nomic-embed-text (d=768) to validate
the approach before integrating into the knowledge store.

Usage:
  python turboquant_vector_search.py
  python turboquant_vector_search.py --n-vectors 10000 --n-queries 100
"""

import argparse
import sys
import time

import torch
import numpy as np

sys.path.insert(0, "/home/ava/dev")
from turboquant_pytorch import TurboQuantProd, TurboQuantMSE


def generate_realistic_vectors(n: int, d: int, seed: int = 42) -> torch.Tensor:
    """Generate vectors mimicking embedding distribution (unit-normalized, clustered)."""
    torch.manual_seed(seed)
    # Mix of clusters to simulate real embedding space
    n_clusters = 20
    cluster_size = n // n_clusters
    vectors = []
    for i in range(n_clusters):
        center = torch.randn(d)
        center = center / center.norm()
        noise = torch.randn(cluster_size, d) * 0.3
        cluster = center.unsqueeze(0) + noise
        # L2 normalize
        cluster = cluster / cluster.norm(dim=1, keepdim=True)
        vectors.append(cluster)
    vectors = torch.cat(vectors, dim=0)[:n]
    return vectors


def brute_force_search(
    queries: torch.Tensor, database: torch.Tensor, k: int
) -> torch.Tensor:
    """Standard brute-force cosine similarity search."""
    # queries: (n_queries, d), database: (n_db, d)
    scores = queries @ database.T  # (n_queries, n_db)
    _, indices = scores.topk(k, dim=1)
    return indices


def turboquant_search(
    queries: torch.Tensor,
    compressed_db: dict,
    quantizer: TurboQuantProd,
    k: int,
) -> torch.Tensor:
    """Search using TurboQuant compressed vectors (batched)."""
    n_queries = queries.shape[0]
    n_db = compressed_db["mse_indices"].shape[0]

    all_scores = torch.zeros(n_queries, n_db)
    for qi in range(n_queries):
        query = queries[qi:qi+1].expand(n_db, -1)
        scores = quantizer.inner_product(query, compressed_db)
        all_scores[qi] = scores

    _, indices = all_scores.topk(k, dim=1)
    return indices


def recall_at_k(gt_indices: torch.Tensor, pred_indices: torch.Tensor, k: int) -> float:
    """Compute recall@k: fraction of ground truth top-k found in predicted top-k."""
    recalls = []
    for i in range(gt_indices.shape[0]):
        gt_set = set(gt_indices[i, :k].tolist())
        pred_set = set(pred_indices[i, :k].tolist())
        recalls.append(len(gt_set & pred_set) / k)
    return np.mean(recalls)


def storage_bytes(n_vectors: int, d: int, bits: int) -> dict:
    """Calculate storage for different representations."""
    fp32 = n_vectors * d * 4
    fp16 = n_vectors * d * 2
    # TurboQuant: (bits-1)*d bits for MSE + d bits for QJL + 16 bits for norm per vector
    tq_bits_per_vec = (bits - 1) * d + d + 16
    tq = n_vectors * tq_bits_per_vec // 8
    return {
        "fp32_bytes": fp32,
        "fp16_bytes": fp16,
        "turboquant_bytes": tq,
        "compression_vs_fp32": fp32 / tq,
        "compression_vs_fp16": fp16 / tq,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=5000)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--dim", type=int, default=768, help="nomic-embed-text = 768")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    d = args.dim
    n_db = args.n_vectors
    n_queries = args.n_queries
    device = args.device

    print(f"TurboQuant Vector Search Experiment")
    print(f"  Vectors: {n_db}, Queries: {n_queries}, Dim: {d}")
    print(f"  Device: {device}")
    print()

    # Generate data
    print("Generating vectors...")
    db_vectors = generate_realistic_vectors(n_db, d, seed=42).to(device)
    query_vectors = generate_realistic_vectors(n_queries, d, seed=123).to(device)

    # Ground truth: brute force
    print("Computing ground truth (brute force)...")
    t0 = time.time()
    gt_k10 = brute_force_search(query_vectors, db_vectors, k=10)
    bf_time = time.time() - t0
    print(f"  Brute force: {bf_time*1000:.1f}ms for {n_queries} queries")

    # Test each bit-width
    for bits in [4, 3, 2]:
        print(f"\n{'='*50}")
        print(f"  TurboQuant {bits}-bit")
        print(f"{'='*50}")

        # Initialize quantizer
        t0 = time.time()
        quantizer = TurboQuantProd(d, bits=bits, seed=42, device=device)
        init_time = time.time() - t0
        print(f"  Init time: {init_time*1000:.1f}ms")

        # Compress database (single batch)
        t0 = time.time()
        compressed_db = quantizer.quantize(db_vectors)
        compress_time = time.time() - t0
        print(f"  Compress time: {compress_time*1000:.1f}ms ({n_db} vectors)")

        # Search
        t0 = time.time()
        pred_k10 = turboquant_search(query_vectors, compressed_db, quantizer, k=10)
        search_time = time.time() - t0
        print(f"  Search time: {search_time*1000:.1f}ms ({n_queries} queries)")

        # Recall
        for k in [1, 5, 10]:
            r = recall_at_k(gt_k10, pred_k10, k)
            print(f"  Recall@{k}: {r:.4f} ({r*100:.1f}%)")

        # Storage
        stor = storage_bytes(n_db, d, bits)
        print(f"  Storage: {stor['turboquant_bytes']/1024:.1f} KB "
              f"(vs {stor['fp32_bytes']/1024:.1f} KB fp32, "
              f"{stor['compression_vs_fp32']:.1f}x compression)")

    print(f"\n{'='*50}")
    print("Storage comparison for protoResearcher scale:")
    for n in [1000, 10000, 100000]:
        s = storage_bytes(n, d, 3)
        print(f"  {n:>7} vectors: fp32={s['fp32_bytes']/1024/1024:.1f}MB, "
              f"TQ3={s['turboquant_bytes']/1024/1024:.1f}MB "
              f"({s['compression_vs_fp32']:.1f}x)")


if __name__ == "__main__":
    main()
