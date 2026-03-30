"""Tool implementations for Context-1 retrieval agent.

Provides the 4 tools from the Context-1 paper:
  - search_corpus: hybrid BM25 + dense vector search
  - grep_corpus: regex search over chunks
  - read_document: retrieve full document by ID
  - prune_chunks: remove chunks from context

Backed by a simple document store (sqlite-vec or in-memory)
compatible with protoResearcher's knowledge base.
"""

from __future__ import annotations

import json
import re
import struct
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .format import ToolDef
except ImportError:
    from format import ToolDef

# --- Tool Definitions (for prompt) ---

TOOL_DEFS = [
    ToolDef(
        name="search_corpus",
        description="Hybrid BM25 + dense vector search. Returns top matching chunks within token budget.",
        params={
            "query": {"type": "string", "desc": "Search query"},
        },
    ),
    ToolDef(
        name="grep_corpus",
        description="Regex search over the corpus. Returns up to 5 matching chunks.",
        params={
            "pattern": {"type": "string", "desc": "Regex pattern to search for"},
        },
    ),
    ToolDef(
        name="read_document",
        description="Read full content of a document by ID. Content truncated to fit token budget.",
        params={
            "doc_id": {"type": "string", "desc": "Document ID to retrieve"},
        },
    ),
    ToolDef(
        name="prune_chunks",
        description="Remove specified chunks from conversation context to free token budget.",
        params={
            "chunk_ids": {"type": "string[]", "desc": "List of chunk IDs to remove"},
        },
    ),
]


@dataclass
class Chunk:
    """A document chunk with metadata."""

    chunk_id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (4 chars per token)."""
        return len(self.content) // 4


class DocumentStore:
    """In-memory document store for Context-1 experiments.

    Supports loading from:
    - Raw text files (one doc per file)
    - protoResearcher sqlite-vec database
    - JSON corpus files

    Search modes:
    - keyword: BM25-lite term frequency (default, no setup needed)
    - dense: FAISS IndexFlatIP with sentence-transformers embeddings
    - hybrid: RRF fusion of keyword + dense results
    - hybrid+rerank: hybrid retrieval + cross-encoder reranking
    """

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}  # doc_id -> full content
        self.chunks: dict[str, Chunk] = {}  # chunk_id -> Chunk
        self._chunk_list: list[str] = []  # ordered chunk IDs for FAISS alignment
        self._faiss_index = None
        self._embed_model = None
        self._reranker = None

    def build_dense_index(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda:1",
        batch_size: int = 2048,
    ) -> None:
        """Build FAISS dense vector index from all chunks. One-time cost."""
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import time

        print(f"Building dense index: {model_name} on {device}...")
        self._embed_model = SentenceTransformer(model_name, device=device)

        # Stable ordering for FAISS alignment
        self._chunk_list = list(self.chunks.keys())
        texts = [self.chunks[cid].content for cid in self._chunk_list]

        t0 = time.perf_counter()
        embeddings = self._embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embed_time = time.perf_counter() - t0
        print(f"Embedded {len(texts)} chunks in {embed_time:.1f}s ({len(texts)/embed_time:.0f}/s)")

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings.astype(np.float32))
        print(f"FAISS index: {self._faiss_index.ntotal} vectors, dim={dim}")

    def save_dense_index(self, path: Path) -> None:
        """Save FAISS index and chunk list to disk for reuse."""
        import faiss
        if self._faiss_index is None:
            raise ValueError("No dense index to save")
        faiss.write_index(self._faiss_index, str(path))
        # Save chunk list alongside
        chunk_list_path = path.with_suffix(".chunks.json")
        chunk_list_path.write_text(json.dumps(self._chunk_list))
        print(f"Saved index to {path} + {chunk_list_path}")

    def load_dense_index(
        self, path: Path, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"
    ) -> None:
        """Load a pre-built FAISS index from disk."""
        import faiss
        from sentence_transformers import SentenceTransformer

        self._faiss_index = faiss.read_index(str(path))
        chunk_list_path = path.with_suffix(".chunks.json")
        self._chunk_list = json.loads(chunk_list_path.read_text())
        self._embed_model = SentenceTransformer(model_name, device=device)
        print(f"Loaded index: {self._faiss_index.ntotal} vectors from {path}")

    def add_document(self, doc_id: str, content: str, chunk_size: int = 512) -> None:
        """Add a document, automatically chunking it."""
        self.documents[doc_id] = content
        # Simple sentence-aware chunking
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            if current_len + len(sentence) > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                cid = f"{doc_id}::chunk_{chunk_idx}"
                self.chunks[cid] = Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    content=chunk_text,
                )
                chunk_idx += 1
                current_chunk = []
                current_len = 0
            current_chunk.append(sentence)
            current_len += len(sentence)

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            cid = f"{doc_id}::chunk_{chunk_idx}"
            self.chunks[cid] = Chunk(chunk_id=cid, doc_id=doc_id, content=chunk_text)

    def load_from_directory(self, path: Path, glob: str = "*.txt") -> int:
        """Load documents from a directory of text files."""
        count = 0
        for f in sorted(path.glob(glob)):
            doc_id = f.stem
            self.add_document(doc_id, f.read_text())
            count += 1
        return count

    def load_from_json(self, path: Path) -> int:
        """Load from JSON file: list of {id, content, metadata?}."""
        data = json.loads(path.read_text())
        for doc in data:
            self.add_document(doc["id"], doc["content"])
        return len(data)

    def load_from_protoresearcher(self, db_path: Path) -> int:
        """Load papers and findings from a protoResearcher sqlite DB."""
        db = sqlite3.connect(str(db_path))
        count = 0

        # Load papers
        for row in db.execute("SELECT id, title, abstract, summary FROM papers").fetchall():
            pid, title, abstract, summary = row
            content = f"# {title}\n\n{abstract or ''}\n\n{summary or ''}"
            self.add_document(f"paper:{pid}", content.strip())
            count += 1

        # Load findings
        for row in db.execute("SELECT id, content, source, topic FROM findings").fetchall():
            fid, content, source, topic = row
            full = f"[{topic or 'general'}] {content}"
            if source:
                full += f"\n(Source: {source})"
            self.add_document(f"finding:{fid}", full)
            count += 1

        # Load digests
        for row in db.execute("SELECT id, title, content FROM digests").fetchall():
            did, title, content = row
            self.add_document(f"digest:{did}", f"# {title}\n\n{content}")
            count += 1

        db.close()
        return count

    def search(
        self, query: str, k: int = 10, exclude_ids: set[str] | None = None
    ) -> list[Chunk]:
        """Hybrid BM25-style keyword + optional vector search."""
        exclude = exclude_ids or set()
        query_terms = set(query.lower().split())

        scored: list[tuple[float, Chunk]] = []
        for cid, chunk in self.chunks.items():
            if cid in exclude:
                continue
            # BM25-lite: term frequency scoring
            words = chunk.content.lower().split()
            word_set = set(words)
            tf_score = sum(1 for t in query_terms if t in word_set)
            if tf_score > 0:
                # Normalize by doc length (simple BM25 approx)
                score = tf_score / (len(words) + 50)
                scored.append((score, chunk))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:k]]

    def dense_search(
        self, query: str, k: int = 20, exclude_ids: set[str] | None = None
    ) -> list[Chunk]:
        """Dense vector search via FAISS. Requires build_dense_index() first."""
        if self._faiss_index is None or self._embed_model is None:
            return self.search(query, k, exclude_ids)  # fallback to keyword

        import numpy as np

        exclude = exclude_ids or set()
        q_emb = self._embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        # Fetch extra to compensate for exclusions
        fetch_k = k + len(exclude)
        scores, indices = self._faiss_index.search(q_emb.astype(np.float32), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            cid = self._chunk_list[idx]
            if cid in exclude:
                continue
            results.append(self.chunks[cid])
            if len(results) >= k:
                break
        return results

    def hybrid_search(
        self, query: str, k: int = 20, exclude_ids: set[str] | None = None
    ) -> list[Chunk]:
        """Hybrid search: RRF fusion of keyword + dense results."""
        keyword_results = self.search(query, k=k * 2, exclude_ids=exclude_ids)
        dense_results = self.dense_search(query, k=k * 2, exclude_ids=exclude_ids)

        # Reciprocal Rank Fusion (RRF) with k=60
        rrf_k = 60
        scores: dict[str, float] = {}
        for rank, chunk in enumerate(keyword_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, chunk in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score
        all_chunks = {c.chunk_id: c for c in keyword_results + dense_results}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [all_chunks[cid] for cid, _ in ranked[:k]]

    def load_reranker(self, model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1") -> None:
        """Load a cross-encoder reranker for result reranking."""
        from sentence_transformers import CrossEncoder
        import time

        print(f"Loading reranker: {model_name}...")
        t0 = time.perf_counter()
        self._reranker = CrossEncoder(model_name, max_length=512, device="cpu")
        print(f"Reranker loaded in {time.perf_counter() - t0:.1f}s")

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
        """Rerank chunks using cross-encoder. Falls back to input order if no reranker."""
        if self._reranker is None:
            return chunks[:top_k]

        pairs = [[query, c.content] for c in chunks]
        scores = self._reranker.predict(pairs, batch_size=len(pairs))
        scored = sorted(zip(chunks, scores), key=lambda x: -x[1])
        return [c for c, _ in scored[:top_k]]

    def hybrid_rerank_search(
        self, query: str, k: int = 10, candidates: int = 40,
        exclude_ids: set[str] | None = None,
    ) -> list[Chunk]:
        """Hybrid retrieval + cross-encoder reranking pipeline.

        1. Retrieve top `candidates` via hybrid search (RRF of keyword + dense)
        2. Rerank with cross-encoder
        3. Return top `k`
        """
        raw = self.hybrid_search(query, k=candidates, exclude_ids=exclude_ids)
        return self.rerank(query, raw, top_k=k)

    def grep(self, pattern: str, max_results: int = 5) -> list[Chunk]:
        """Regex search over all chunks."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []

        results = []
        for chunk in self.chunks.values():
            if regex.search(chunk.content):
                results.append(chunk)
                if len(results) >= max_results:
                    break
        return results

    def read_document(self, doc_id: str, max_chars: int = 8000) -> str | None:
        """Read full document content, truncated to budget."""
        content = self.documents.get(doc_id)
        if content and len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated]"
        return content


class ToolExecutor:
    """Executes tool calls against a DocumentStore."""

    def __init__(self, store: DocumentStore, token_budget: int = 24000) -> None:
        self.store = store
        self.token_budget = token_budget
        self.seen_chunk_ids: set[str] = set()
        self.active_chunks: dict[str, Chunk] = {}  # chunks in context
        self._token_count = 0

    @property
    def token_usage(self) -> tuple[int, int]:
        """Current (used, budget) token counts."""
        chunk_tokens = sum(c.token_estimate for c in self.active_chunks.values())
        return (self._token_count + chunk_tokens, self.token_budget)

    def update_token_count(self, prompt_tokens: int) -> None:
        """Update the base token count (prompt without chunks)."""
        self._token_count = prompt_tokens

    def execute(self, tool_name: str, arguments: str) -> str:
        """Execute a tool call and return the result string."""
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

        if tool_name == "search_corpus":
            return self._search(args.get("query", ""))
        elif tool_name == "grep_corpus":
            return self._grep(args.get("pattern", ""))
        elif tool_name == "read_document":
            return self._read(args.get("doc_id", ""))
        elif tool_name == "prune_chunks":
            return self._prune(args.get("chunk_ids", []))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _search(self, query: str) -> str:
        chunks = self.store.search(query, k=10, exclude_ids=self.seen_chunk_ids)
        if not chunks:
            return json.dumps({"results": [], "message": "No results found."})

        results = []
        for chunk in chunks:
            self.seen_chunk_ids.add(chunk.chunk_id)
            self.active_chunks[chunk.chunk_id] = chunk
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
            })

        return json.dumps({"results": results, "count": len(results)})

    def _grep(self, pattern: str) -> str:
        chunks = self.store.grep(pattern, max_results=5)
        if not chunks:
            return json.dumps({"results": [], "message": "No matches."})

        results = []
        for chunk in chunks:
            self.seen_chunk_ids.add(chunk.chunk_id)
            self.active_chunks[chunk.chunk_id] = chunk
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
            })

        return json.dumps({"results": results, "count": len(results)})

    def _read(self, doc_id: str) -> str:
        content = self.store.read_document(doc_id)
        if content is None:
            return json.dumps({"error": f"Document not found: {doc_id}"})
        return json.dumps({"doc_id": doc_id, "content": content})

    def _prune(self, chunk_ids: list[str]) -> str:
        pruned = 0
        for cid in chunk_ids:
            if cid in self.active_chunks:
                del self.active_chunks[cid]
                pruned += 1
        return json.dumps({"pruned": pruned, "remaining": len(self.active_chunks)})
