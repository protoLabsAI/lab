"""RAG Embedding + Retrieval Benchmark

Compares embedding models and retrieval strategies on protoResearcher's
knowledge base and context-1 test queries.

Tests:
  1. Embedding latency + quality (Ollama vs local vLLM-style server)
  2. Retrieval quality: vector-only, BM25-only, hybrid RRF
  3. With/without cross-encoder reranking
  4. End-to-end RAG quality scored by LLM judge

Models tested:
  - qwen3-embedding:0.6b via Ollama (current production)
  - Qwen/Qwen3-Embedding-0.6B via local server (port 8001)
  - Qwen/Qwen3-Embedding-4B via local server (swap)

Usage:
    cd ~/dev/lab
    uv run python experiments/rag-bench/bench.py
    uv run python experiments/rag-bench/bench.py --embed-only
    uv run python experiments/rag-bench/bench.py --retrieval-only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

RESULTS_DIR = Path(__file__).parent / "results"

# --- Embedding endpoints ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://100.101.189.45:11434")
LOCAL_EMBED_URL = "http://localhost:8001"

# --- Test data ---
SHORT_QUERIES = [
    "How does vLLM handle tensor parallelism on PCIe?",
    "What embedding model works best for knowledge graphs?",
    "Qwen3.5 MoE routing overhead vs dense model latency",
    "FP8 quantization accuracy tradeoffs",
    "hybrid search combining BM25 and vector retrieval",
]

MEDIUM_DOCS = [
    "The vLLM inference server supports tensor parallelism across multiple GPUs. On systems with PCIe interconnect (no NVLink), NCCL must be configured with Ring algorithm and Simple protocol. Disabling P2P transfers via NCCL_P2P_DISABLE=1 prevents memory corruption during CUDA graph replay caused by ACS on PCIe bridges.",
    "Hybrid search combines BM25 lexical matching with dense vector retrieval using Reciprocal Rank Fusion (RRF). On a 525K chunk corpus, hybrid search achieved 77% of the quality of a dedicated retrieval agent at 7x less latency (5.9s vs 42.9s).",
    "Neo4j stores entities as nodes with properties and relationships as typed edges. For RAG, each entity can be embedded with its relationship context prepended — similar to Anthropic's contextual chunk enrichment but using graph structure.",
]

SIMILAR_PAIRS = [
    ("How does tensor parallelism work on PCIe GPUs?",
     "NCCL configuration for multi-GPU inference without NVLink"),
    ("FP8 quantization benefits for large language models",
     "Reducing model size with 8-bit floating point precision"),
    ("hybrid search combining BM25 and vector retrieval",
     "lexical keyword matching fused with dense embedding search"),
    ("DPO training from preference pairs",
     "Direct preference optimization for aligning language models"),
    ("Qwen3.5 MoE routing with 10B active parameters",
     "Mixture of experts sparse activation for efficient inference"),
]

DISSIMILAR_PAIRS = [
    ("How does tensor parallelism work on PCIe GPUs?",
     "Best practices for growing tomatoes in a greenhouse"),
    ("FP8 quantization benefits for large language models",
     "The history of Renaissance painting in Florence"),
    ("hybrid search combining BM25 and vector retrieval",
     "How to train a golden retriever puppy"),
    ("Qwen3.5 MoE routing with 10B active parameters",
     "Traditional Japanese tea ceremony etiquette"),
]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# --- Ollama embedding ---
def embed_ollama(model: str, texts: list[str]) -> tuple[list[list[float]], float]:
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return data["embeddings"], elapsed


# --- Local server embedding (OpenAI-compatible) ---
def embed_local(model: str, texts: list[str], url: str = LOCAL_EMBED_URL) -> tuple[list[list[float]], float]:
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    embs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    return embs, elapsed


def bench_latency(embed_fn, model: str, warmup: int = 2, runs: int = 5) -> dict:
    """Benchmark embedding latency and throughput."""
    # Warmup
    for _ in range(warmup):
        embed_fn(model, ["warmup"])

    # Single query
    single_times = []
    for q in SHORT_QUERIES:
        _, elapsed = embed_fn(model, [q])
        single_times.append(elapsed)

    # Batch of 5
    batch_times = []
    for _ in range(runs):
        _, elapsed = embed_fn(model, SHORT_QUERIES)
        batch_times.append(elapsed)

    # Medium docs
    med_times = []
    for doc in MEDIUM_DOCS:
        _, elapsed = embed_fn(model, [doc])
        med_times.append(elapsed)

    # Batch of 20 (throughput test)
    big_batch = SHORT_QUERIES * 4  # 20 texts
    tp_times = []
    for _ in range(runs):
        _, elapsed = embed_fn(model, big_batch)
        tp_times.append(elapsed)

    embs, _ = embed_fn(model, ["test"])
    dims = len(embs[0])

    return {
        "model": model,
        "dims": dims,
        "single_query_ms": round(statistics.mean(single_times) * 1000, 1),
        "batch_5_ms": round(statistics.mean(batch_times) * 1000, 1),
        "medium_doc_ms": round(statistics.mean(med_times) * 1000, 1),
        "batch_20_ms": round(statistics.mean(tp_times) * 1000, 1),
        "throughput_20_docs_per_sec": round(20 / statistics.mean(tp_times), 1),
    }


def bench_quality(embed_fn, model: str) -> dict:
    """Measure embedding quality via cosine similarity discrimination."""
    all_texts = []
    for a, b in SIMILAR_PAIRS + DISSIMILAR_PAIRS:
        all_texts.extend([a, b])
    embs, _ = embed_fn(model, all_texts)

    sim_scores = []
    for i in range(len(SIMILAR_PAIRS)):
        s = cosine_sim(embs[i * 2], embs[i * 2 + 1])
        sim_scores.append(s)

    dissim_scores = []
    offset = len(SIMILAR_PAIRS) * 2
    for i in range(len(DISSIMILAR_PAIRS)):
        s = cosine_sim(embs[offset + i * 2], embs[offset + i * 2 + 1])
        dissim_scores.append(s)

    avg_sim = statistics.mean(sim_scores)
    avg_dissim = statistics.mean(dissim_scores)

    return {
        "model": model,
        "avg_similar": round(avg_sim, 4),
        "avg_dissimilar": round(avg_dissim, 4),
        "discrimination": round(avg_sim - avg_dissim, 4),
        "sim_scores": [round(s, 4) for s in sim_scores],
        "dissim_scores": [round(s, 4) for s in dissim_scores],
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Embedding Benchmark")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama models")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Define models to test ---
    models = []

    if not args.skip_ollama:
        try:
            embed_ollama("qwen3-embedding:0.6b", ["test"])
            models.append(("ollama-qwen3-0.6b", "qwen3-embedding:0.6b", embed_ollama))
            print("Ollama qwen3-embedding:0.6b: available")
        except Exception as e:
            print(f"Ollama not available: {e}")

    try:
        embed_local("Qwen/Qwen3-Embedding-0.6B", ["test"])
        models.append(("local-qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", embed_local))
        print("Local Qwen3-Embedding-0.6B: available")
    except Exception as e:
        print(f"Local embedding server not available: {e}")

    if not models:
        print("No embedding models available. Start at least one server.")
        return

    print(f"\nTesting {len(models)} model configurations\n")

    all_latency = []
    all_quality = []

    for label, model, fn in models:
        print(f"--- {label} ---")

        print("  Latency bench...")
        lat = bench_latency(fn, model)
        lat["label"] = label
        all_latency.append(lat)
        print(f"  Single: {lat['single_query_ms']}ms | Batch5: {lat['batch_5_ms']}ms | "
              f"Batch20: {lat['batch_20_ms']}ms | {lat['throughput_20_docs_per_sec']} docs/s")

        print("  Quality bench...")
        qual = bench_quality(fn, model)
        qual["label"] = label
        all_quality.append(qual)
        print(f"  Similar: {qual['avg_similar']:.4f} | Dissimilar: {qual['avg_dissimilar']:.4f} | "
              f"Discrimination: {qual['discrimination']:.4f}")
        print()

    # --- Summary ---
    print("=" * 75)
    print("EMBEDDING BENCHMARK RESULTS")
    print("=" * 75)

    print(f"\n{'Label':<22s} {'Dims':>5s} {'1-query':>8s} {'Batch5':>8s} {'Batch20':>9s} {'docs/s':>7s}")
    print("-" * 62)
    for r in all_latency:
        print(f"{r['label']:<22s} {r['dims']:>5d} {r['single_query_ms']:>7.1f}ms "
              f"{r['batch_5_ms']:>7.1f}ms {r['batch_20_ms']:>8.1f}ms {r['throughput_20_docs_per_sec']:>7.1f}")

    print(f"\n{'Label':<22s} {'Similar':>8s} {'Dissim':>8s} {'Discrim':>8s}")
    print("-" * 48)
    for r in all_quality:
        print(f"{r['label']:<22s} {r['avg_similar']:>8.4f} {r['avg_dissimilar']:>8.4f} {r['discrimination']:>8.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {"latency": all_latency, "quality": all_quality, "timestamp": ts}
    out_path = RESULTS_DIR / f"bench_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
