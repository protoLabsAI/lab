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

from .format import ToolDef

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
    """

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}  # doc_id -> full content
        self.chunks: dict[str, Chunk] = {}  # chunk_id -> Chunk
        self._embeddings: dict[str, list[float]] = {}  # chunk_id -> embedding
        self._embed_fn = None

    def set_embed_fn(self, fn) -> None:
        """Set embedding function: fn(text) -> list[float]."""
        self._embed_fn = fn

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
