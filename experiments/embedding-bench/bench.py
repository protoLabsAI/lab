"""Embedding model benchmark: nomic-embed-text vs qwen3-embedding (0.6B, 8B)

Measures: latency, throughput, dimensionality, and cosine similarity quality
on graph-enriched RAG-style inputs relevant to the protoLabs pipeline.
"""

import json
import time
import statistics
import urllib.request
from itertools import combinations

OLLAMA_URL = "http://localhost:11434/api/embed"

MODELS = [
    "nomic-embed-text",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:8b",
]

# Realistic inputs for a Neo4j graph-enriched RAG pipeline
# Mix of short queries, medium docs, and long entity-enriched chunks
TEST_INPUTS = {
    "short_query": [
        "How does vLLM handle tensor parallelism on PCIe?",
        "What embedding model works best for knowledge graphs?",
        "Qwen3.5 MoE routing overhead vs dense model latency",
        "secrets management across Tailscale nodes",
        "FP8 quantization accuracy tradeoffs",
    ],
    "medium_doc": [
        "The vLLM inference server supports tensor parallelism across multiple GPUs. On systems with PCIe interconnect (no NVLink), NCCL must be configured with Ring algorithm and Simple protocol. Disabling P2P transfers via NCCL_P2P_DISABLE=1 prevents memory corruption during CUDA graph replay caused by ACS on PCIe bridges.",
        "Hybrid search combines BM25 lexical matching with dense vector retrieval using Reciprocal Rank Fusion (RRF). On a 525K chunk corpus, hybrid search achieved 77% of the quality of a dedicated retrieval agent at 7x less latency (5.9s vs 42.9s). The key insight is that keywords catch exact terms that vectors miss, and vectors catch semantic similarity that keywords miss.",
        "Neo4j stores entities as nodes with properties and relationships as typed edges. For RAG, each entity can be embedded with its relationship context prepended — similar to Anthropic's contextual chunk enrichment but using graph structure instead of document hierarchy. This creates graph-enriched embeddings that capture both the entity's content and its position in the knowledge graph.",
    ],
    "long_enriched": [
        # Simulates a graph-enriched entity embedding (entity + properties + relationships)
        """Entity: Qwen3.5-122B-A10B-FP8
Type: Language Model
Properties: architecture=MoE, total_params=122B, active_params=10B, precision=FP8, context_window=65536, throughput=112tok/s
Relationships:
- RUNS_ON -> protolabs (AI inference node, RTX PRO 6000 Blackwell x2, 192GB VRAM)
- SERVED_BY -> vLLM (inference server, tensor_parallel=2, port=8000)
- QUANTIZED_FROM -> Qwen3.5-122B (base model, BF16, 244GB)
- EVALUATED_BY -> claw-eval (agentic tool use, 52 tasks), quick-profile (10 suites)
- OUTPERFORMS -> Qwen3.5-27B (dense, 53tok/s, lower quality ceiling)
- SAME_FAMILY -> Qwen3.5-35B-A3B (MoE, 3B active, 180tok/s, speed king)
- STORED_AT -> /mnt/models/huggingface (Samsung 9100 PRO 1TB NVMe)
- PUBLISHED_TO -> HuggingFace/protoLabsAI (FP8 quantized weights)
Summary: Quality ceiling model for the protoLabs stack. Uses TP=2 across both Blackwell GPUs with NCCL_P2P_DISABLE=1 for stable CUDA graphs. FP8 official weights from Qwen — no custom quantization needed. Achieves 112 tok/s decode, 29ms TTFT with prefix caching after warmup.""",
    ],
}

# Semantic similarity pairs (should be high similarity)
SIMILAR_PAIRS = [
    ("How does tensor parallelism work on PCIe GPUs?",
     "NCCL configuration for multi-GPU inference without NVLink"),
    ("FP8 quantization benefits for large language models",
     "Reducing model size with 8-bit floating point precision"),
    ("hybrid search combining BM25 and vector retrieval",
     "lexical keyword matching fused with dense embedding search"),
]

# Dissimilar pairs (should be low similarity)
DISSIMILAR_PAIRS = [
    ("How does tensor parallelism work on PCIe GPUs?",
     "Best practices for growing tomatoes in a greenhouse"),
    ("FP8 quantization benefits for large language models",
     "The history of Renaissance painting in Florence"),
]


def embed(model: str, inputs: list[str]) -> dict:
    """Call Ollama embed API, return response with timing."""
    payload = json.dumps({"model": model, "input": inputs}).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    data["_elapsed"] = elapsed
    return data


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def run_latency_bench(model: str, warmup: int = 2, runs: int = 5) -> dict:
    """Benchmark single-input and batch latency."""
    # Warmup
    for _ in range(warmup):
        embed(model, ["warmup"])

    # Single input latency
    single_times = []
    for q in TEST_INPUTS["short_query"]:
        r = embed(model, [q])
        single_times.append(r["_elapsed"])

    # Batch latency (all short queries at once)
    batch_times = []
    for _ in range(runs):
        r = embed(model, TEST_INPUTS["short_query"])
        batch_times.append(r["_elapsed"])

    # Medium doc latency
    med_times = []
    for doc in TEST_INPUTS["medium_doc"]:
        r = embed(model, [doc])
        med_times.append(r["_elapsed"])

    # Long enriched entity
    long_times = []
    for _ in range(runs):
        r = embed(model, TEST_INPUTS["long_enriched"])
        long_times.append(r["_elapsed"])

    # Get dims
    r = embed(model, ["test"])
    dims = len(r["embeddings"][0])

    return {
        "model": model,
        "dims": dims,
        "single_query_ms": round(statistics.mean(single_times) * 1000, 1),
        "batch_5_ms": round(statistics.mean(batch_times) * 1000, 1),
        "medium_doc_ms": round(statistics.mean(med_times) * 1000, 1),
        "long_entity_ms": round(statistics.mean(long_times) * 1000, 1),
        "throughput_docs_per_sec": round(5 / statistics.mean(batch_times), 1),
    }


def run_quality_bench(model: str) -> dict:
    """Measure embedding quality via cosine similarity on known pairs."""
    # Embed all similarity test texts
    all_texts = []
    for a, b in SIMILAR_PAIRS + DISSIMILAR_PAIRS:
        all_texts.extend([a, b])
    r = embed(model, all_texts)
    embs = r["embeddings"]

    sim_scores = []
    for i in range(len(SIMILAR_PAIRS)):
        s = cosine_sim(embs[i * 2], embs[i * 2 + 1])
        sim_scores.append(s)

    dissim_scores = []
    offset = len(SIMILAR_PAIRS) * 2
    for i in range(len(DISSIMILAR_PAIRS)):
        s = cosine_sim(embs[offset + i * 2], embs[offset + i * 2 + 1])
        dissim_scores.append(s)

    # Cross-category discrimination: avg(similar) - avg(dissimilar)
    avg_sim = statistics.mean(sim_scores)
    avg_dissim = statistics.mean(dissim_scores)

    return {
        "model": model,
        "avg_similar_cosine": round(avg_sim, 4),
        "avg_dissimilar_cosine": round(avg_dissim, 4),
        "discrimination": round(avg_sim - avg_dissim, 4),
        "similar_scores": [round(s, 4) for s in sim_scores],
        "dissimilar_scores": [round(s, 4) for s in dissim_scores],
    }


def main():
    print("=" * 70)
    print("  Embedding Model Benchmark — protoLabs")
    print("=" * 70)

    latency_results = []
    quality_results = []

    for model in MODELS:
        print(f"\n--- {model} ---")
        print("  Running latency bench...")
        lat = run_latency_bench(model)
        latency_results.append(lat)
        print(f"  Dims: {lat['dims']} | Single: {lat['single_query_ms']}ms | "
              f"Batch(5): {lat['batch_5_ms']}ms | "
              f"Medium: {lat['medium_doc_ms']}ms | "
              f"Long entity: {lat['long_entity_ms']}ms | "
              f"Throughput: {lat['throughput_docs_per_sec']} docs/s")

        print("  Running quality bench...")
        qual = run_quality_bench(model)
        quality_results.append(qual)
        print(f"  Similar: {qual['avg_similar_cosine']} | "
              f"Dissimilar: {qual['avg_dissimilar_cosine']} | "
              f"Discrimination: {qual['discrimination']}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Model':<25s} {'Dims':>5s} {'Query':>7s} {'Batch5':>8s} {'MedDoc':>8s} {'Entity':>8s} {'doc/s':>7s}")
    print("-" * 70)
    for r in latency_results:
        print(f"{r['model']:<25s} {r['dims']:>5d} {r['single_query_ms']:>6.1f}ms {r['batch_5_ms']:>6.1f}ms "
              f"{r['medium_doc_ms']:>6.1f}ms {r['long_entity_ms']:>6.1f}ms {r['throughput_docs_per_sec']:>7.1f}")

    print(f"\n{'Model':<25s} {'Similar':>8s} {'Dissim':>8s} {'Discrim':>8s}")
    print("-" * 52)
    for r in quality_results:
        print(f"{r['model']:<25s} {r['avg_similar_cosine']:>8.4f} {r['avg_dissimilar_cosine']:>8.4f} {r['discrimination']:>8.4f}")

    # Save results
    results = {"latency": latency_results, "quality": quality_results}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json")


if __name__ == "__main__":
    main()
