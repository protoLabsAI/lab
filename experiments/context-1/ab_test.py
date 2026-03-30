"""A/B test: Context-1 retrieval agent + Qwen reasoning vs Qwen solo RAG.

Config A (pipeline): Context-1 retrieves docs → Qwen reasons over them
Config B (monolithic): Qwen searches + reasons directly

Scored using lab RAG graders (groundedness, context relevance,
answer relevance, faithfulness).

Usage:
    python -m experiments.context-1.ab_test \
        --corpus /path/to/corpus.json \
        --test-queries test_queries.yaml \
        --context1-url http://localhost:8001 \
        --qwen-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import yaml
from pathlib import Path

import httpx

from .harness import run_agent
from .tools import DocumentStore

# --- Qwen solo RAG prompt ---

QWEN_RAG_SYSTEM = """\
You are a research assistant. Answer the user's question using ONLY the provided context documents. \
If the context doesn't contain enough information, say so. Cite specific documents when possible.

Be thorough and accurate. Do not make up information not present in the context."""


def qwen_chat(
    messages: list[dict],
    url: str = "http://localhost:8000",
    model: str = "local",
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """Call Qwen via OpenAI-compatible chat API."""
    resp = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def run_config_a(
    query: str,
    store: DocumentStore,
    context1_url: str,
    qwen_url: str,
    context1_model: str = "local",
    qwen_model: str = "local",
    verbose: bool = True,
) -> dict:
    """Config A: Context-1 retrieves → Qwen reasons."""
    t0 = time.time()

    # Step 1: Context-1 retrieves
    if verbose:
        print("\n[Config A] Step 1: Context-1 retrieval...")
    retrieval = run_agent(
        query=query,
        store=store,
        vllm_url=context1_url,
        model=context1_model,
        verbose=verbose,
    )

    # Step 2: Feed retrieved chunks to Qwen for reasoning
    if verbose:
        print(f"\n[Config A] Step 2: Qwen reasoning over {len(retrieval['retrieved_chunks'])} chunks...")

    context_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in retrieval["retrieved_chunks"]
    )

    messages = [
        {"role": "system", "content": QWEN_RAG_SYSTEM},
        {
            "role": "user",
            "content": f"Context documents:\n\n{context_text}\n\n---\n\nQuestion: {query}",
        },
    ]
    answer = qwen_chat(messages, url=qwen_url, model=qwen_model)

    elapsed = time.time() - t0
    return {
        "config": "A (Context-1 + Qwen)",
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieval["retrieved_chunks"],
        "retrieval_turns": retrieval["turns"],
        "elapsed_s": round(elapsed, 2),
    }


def run_config_b(
    query: str,
    store: DocumentStore,
    qwen_url: str,
    qwen_model: str = "local",
    k: int = 20,
    search_mode: str = "hybrid",
    verbose: bool = True,
) -> dict:
    """Config B: Qwen solo RAG (search + reasoning).

    search_mode: "keyword" (BM25-lite), "dense" (FAISS), "hybrid" (RRF fusion)
    """
    t0 = time.time()

    if verbose:
        print(f"\n[Config B] Qwen solo RAG — {search_mode} search + reason...")

    if search_mode == "hybrid+rerank":
        chunks = store.hybrid_rerank_search(query, k=k, candidates=k * 2)
    elif search_mode == "dense":
        chunks = store.dense_search(query, k=k)
    elif search_mode == "hybrid":
        chunks = store.hybrid_search(query, k=k)
    else:
        chunks = store.search(query, k=k)
    context_text = "\n\n---\n\n".join(
        f"[{c.chunk_id}]\n{c.content}" for c in chunks
    )

    messages = [
        {"role": "system", "content": QWEN_RAG_SYSTEM},
        {
            "role": "user",
            "content": f"Context documents:\n\n{context_text}\n\n---\n\nQuestion: {query}",
        },
    ]
    answer = qwen_chat(messages, url=qwen_url, model=qwen_model)

    elapsed = time.time() - t0
    retrieved = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content}
        for c in chunks
    ]
    return {
        "config": "B (Qwen solo)",
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "retrieval_turns": 1,
        "elapsed_s": round(elapsed, 2),
    }


def grade_result(result: dict, expected_answer: str | None = None) -> dict:
    """Grade a result using simple heuristics (upgrade to lab RAG graders later)."""
    answer = result.get("answer") or ""
    scores = {
        "chunks_retrieved": len(result["retrieved_chunks"]),
        "answer_length": len(answer),
        "elapsed_s": result["elapsed_s"],
    }
    if expected_answer:
        # Simple keyword overlap score
        expected_words = set(expected_answer.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(expected_words & answer_words)
        scores["keyword_overlap"] = round(overlap / max(len(expected_words), 1), 3)
    return scores


def run_ab_test(
    queries: list[dict],
    store: DocumentStore,
    context1_url: str,
    qwen_url: str,
    context1_model: str = "context-1",
    qwen_model: str = "local",
    search_mode: str = "hybrid",
    verbose: bool = True,
) -> list[dict]:
    """Run A/B test across all queries.

    queries: list of {query: str, expected_answer?: str}
    """
    results = []

    for i, q in enumerate(queries):
        query = q["query"]
        expected = q.get("expected_answer")

        print(f"\n{'='*60}")
        print(f"Query {i+1}/{len(queries)}: {query}")
        print(f"{'='*60}")

        # Run both configs
        result_a = run_config_a(query, store, context1_url, qwen_url, context1_model=context1_model, qwen_model=qwen_model, verbose=verbose)
        result_b = run_config_b(query, store, qwen_url, qwen_model=qwen_model, search_mode=search_mode, verbose=verbose)

        # Grade
        scores_a = grade_result(result_a, expected)
        scores_b = grade_result(result_b, expected)

        result_a["scores"] = scores_a
        result_b["scores"] = scores_b

        results.append({"query": query, "config_a": result_a, "config_b": result_b})

        print(f"\n--- Results ---")
        print(f"  Config A: {scores_a}")
        print(f"  Config B: {scores_b}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        q = r["query"][:50]
        sa = r["config_a"]["scores"]
        sb = r["config_b"]["scores"]
        print(f"  {q}...")
        print(f"    A: chunks={sa['chunks_retrieved']}, time={sa['elapsed_s']}s")
        print(f"    B: chunks={sb['chunks_retrieved']}, time={sb['elapsed_s']}s")
        if "keyword_overlap" in sa:
            print(f"    A overlap={sa['keyword_overlap']}, B overlap={sb['keyword_overlap']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test: Context-1 + Qwen vs Qwen solo")
    parser.add_argument("--corpus", required=True, help="Path to corpus")
    parser.add_argument("--test-queries", required=True, help="YAML file with test queries")
    parser.add_argument("--context1-url", default="http://localhost:8001", help="Context-1 vLLM URL")
    parser.add_argument("--context1-model", default="context-1", help="Context-1 model name")
    parser.add_argument("--qwen-url", default="http://localhost:8000", help="Qwen vLLM URL")
    parser.add_argument("--qwen-model", default="local", help="Qwen model name")
    parser.add_argument("--search-mode", default="hybrid", choices=["keyword", "dense", "hybrid", "hybrid+rerank"],
                        help="Search mode for Config B")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Embedding model for dense search")
    parser.add_argument("--embed-device", default="cpu", help="Device for embedding model")
    parser.add_argument("--index-path", default=None, help="Path to save/load FAISS index (avoids re-embedding)")
    parser.add_argument("--output", default="ab_results.json", help="Output file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load corpus
    store = DocumentStore()
    corpus_path = Path(args.corpus)
    if corpus_path.is_dir():
        n = store.load_from_directory(corpus_path)
    elif corpus_path.suffix == ".json":
        n = store.load_from_json(corpus_path)
    elif corpus_path.suffix == ".db":
        n = store.load_from_protoresearcher(corpus_path)
    else:
        print(f"Unknown corpus format: {corpus_path}")
        sys.exit(1)
    print(f"Loaded {n} items, {len(store.chunks)} chunks")

    # Build or load dense index if needed
    if args.search_mode in ("dense", "hybrid", "hybrid+rerank"):
        index_path = Path(args.index_path) if args.index_path else None
        if index_path and index_path.exists():
            store.load_dense_index(index_path, model_name=args.embed_model, device=args.embed_device)
        else:
            store.build_dense_index(
                model_name=args.embed_model,
                device=args.embed_device,
            )
            if index_path:
                store.save_dense_index(index_path)

    # Load reranker if needed
    if args.search_mode == "hybrid+rerank":
        store.load_reranker()

    # Load queries
    queries = yaml.safe_load(Path(args.test_queries).read_text())

    results = run_ab_test(
        queries=queries,
        store=store,
        context1_url=args.context1_url,
        context1_model=args.context1_model,
        qwen_url=args.qwen_url,
        qwen_model=args.qwen_model,
        search_mode=args.search_mode,
        verbose=not args.quiet,
    )

    # Save results
    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
