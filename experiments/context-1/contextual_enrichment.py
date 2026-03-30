"""Contextual chunk enrichment experiment.

Based on Anthropic's Contextual Retrieval: prepend LLM-generated context
to each chunk before embedding. Tests if enriched chunks improve retrieval.

Usage:
    python experiments/context-1/contextual_enrichment.py \
        --corpus experiments/context-1/large_corpus.json \
        --qwen-url http://localhost:8000 \
        --queries experiments/context-1/hard_queries.yaml
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
from tools import DocumentStore, Chunk

CONTEXT_PROMPT = """\
<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a short succinct context (1-2 sentences) to situate this chunk within the overall document for improving search retrieval. Answer only with the context, nothing else."""

RAG_SYSTEM = """\
You are a research assistant. Answer the user's question using ONLY the provided context documents. \
If the context doesn't contain enough information, say so. Cite specific documents when possible."""


def qwen_chat(messages, url, model="local", max_tokens=200):
    resp = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def enrich_chunk(doc_content: str, chunk_content: str, url: str) -> str:
    """Generate contextual prefix for a chunk using LLM."""
    # Truncate document to fit in context (keep first 3000 chars)
    doc_truncated = doc_content[:3000]
    prompt = CONTEXT_PROMPT.format(document=doc_truncated, chunk=chunk_content)
    try:
        context = qwen_chat(
            [{"role": "user", "content": prompt}],
            url, max_tokens=100,
        )
        return f"{context.strip()} {chunk_content}"
    except Exception as e:
        return chunk_content  # fallback to original


def build_enriched_store(
    corpus_path: Path, qwen_url: str, max_docs: int = 0, sample_chunks: int = 0
) -> tuple[DocumentStore, DocumentStore]:
    """Build both a baseline and enriched store from the same corpus.

    Returns (baseline_store, enriched_store).
    """
    data = json.loads(corpus_path.read_text())
    if max_docs:
        data = data[:max_docs]

    baseline = DocumentStore()
    enriched = DocumentStore()

    for doc in data:
        baseline.add_document(doc["id"], doc["content"])
        enriched.add_document(doc["id"], doc["content"])

    # Now enrich a subset of chunks
    chunk_ids = list(enriched.chunks.keys())
    if sample_chunks:
        chunk_ids = chunk_ids[:sample_chunks]

    print(f"Enriching {len(chunk_ids)} chunks via LLM...")
    t0 = time.time()
    enriched_count = 0

    for i, cid in enumerate(chunk_ids):
        chunk = enriched.chunks[cid]
        doc_content = enriched.documents.get(chunk.doc_id, "")

        new_content = enrich_chunk(doc_content, chunk.content, qwen_url)
        if new_content != chunk.content:
            enriched.chunks[cid] = Chunk(
                chunk_id=cid,
                doc_id=chunk.doc_id,
                content=new_content,
            )
            enriched_count += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(chunk_ids) - i - 1) / rate
            print(f"  {i+1}/{len(chunk_ids)} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"Enriched {enriched_count}/{len(chunk_ids)} chunks in {elapsed:.1f}s")

    return baseline, enriched


def search_and_answer(query, store, qwen_url, k=20):
    """Hybrid search + answer."""
    t0 = time.time()
    chunks = store.hybrid_search(query, k=k)
    context_text = "\n\n---\n\n".join(
        f"[{c.chunk_id}]\n{c.content}" for c in chunks
    )
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Context:\n\n{context_text}\n\n---\n\nQuestion: {query}"},
    ]
    resp = httpx.post(
        f"{qwen_url}/v1/chat/completions",
        json={
            "model": "local", "messages": messages,
            "max_tokens": 2048, "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"] or ""
    elapsed = time.time() - t0
    return {
        "answer": answer,
        "answer_length": len(answer),
        "chunks": len(chunks),
        "total_s": round(elapsed, 2),
        "top_chunk_ids": [c.chunk_id for c in chunks[:5]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qwen-url", default="http://localhost:8000")
    parser.add_argument("--max-docs", type=int, default=200, help="Max docs to process (0=all)")
    parser.add_argument("--sample-chunks", type=int, default=500, help="Max chunks to enrich (0=all)")
    parser.add_argument("--output", default="experiments/context-1/contextual_results.json")
    args = parser.parse_args()

    queries = yaml.safe_load(Path(args.queries).read_text())

    print("Building baseline and enriched stores...")
    baseline, enriched = build_enriched_store(
        Path(args.corpus), args.qwen_url,
        max_docs=args.max_docs, sample_chunks=args.sample_chunks,
    )
    print(f"Baseline: {len(baseline.chunks)} chunks")
    print(f"Enriched: {len(enriched.chunks)} chunks")

    # Build dense indices for both
    print("\nBuilding dense index for baseline...")
    baseline.build_dense_index(device="cpu", batch_size=512)
    print("Building dense index for enriched...")
    enriched.build_dense_index(device="cpu", batch_size=512)

    results = []
    for i, q in enumerate(queries):
        query = q["query"]
        print(f"\nQuery {i+1}/{len(queries)}: {query[:60]}...")

        r_base = search_and_answer(query, baseline, args.qwen_url)
        r_enrich = search_and_answer(query, enriched, args.qwen_url)

        print(f"  Baseline: {r_base['answer_length']} chars, {r_base['total_s']}s")
        print(f"  Enriched: {r_enrich['answer_length']} chars, {r_enrich['total_s']}s")

        # Check if different chunks were retrieved
        overlap = len(set(r_base["top_chunk_ids"]) & set(r_enrich["top_chunk_ids"]))
        print(f"  Top-5 overlap: {overlap}/5")

        results.append({
            "query": query,
            "baseline": r_base,
            "enriched": r_enrich,
            "top5_overlap": overlap,
        })

    # Summary
    n = len(results)
    avg_base = sum(r["baseline"]["answer_length"] for r in results) / n
    avg_enrich = sum(r["enriched"]["answer_length"] for r in results) / n
    avg_overlap = sum(r["top5_overlap"] for r in results) / n
    delta = (avg_enrich - avg_base) / avg_base * 100

    print(f"\n{'='*60}")
    print(f"SUMMARY ({n} queries)")
    print(f"  Baseline avg: {avg_base:.0f} chars")
    print(f"  Enriched avg: {avg_enrich:.0f} chars ({delta:+.1f}%)")
    print(f"  Avg top-5 overlap: {avg_overlap:.1f}/5")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
