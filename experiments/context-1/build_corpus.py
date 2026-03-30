"""Build a large test corpus from local repo documentation.

Harvests markdown, Python docstrings, configs, and READMEs from
the protoLabs ecosystem to create a realistic multi-hop retrieval test.

Usage:
    python -m experiments.context-1.build_corpus --output experiments/context-1/large_corpus.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Directories to harvest
SOURCES = [
    # Lab repo — core docs, evals, experiments, infra
    ("lab", Path.home() / "dev/lab"),
    # protoResearcher — AI agent, knowledge system
    ("protoResearcher", Path.home() / "dev/protoResearcher"),
    # homelab-iac — infrastructure as code
    ("homelab-iac", Path.home() / "dev/homelab-iac"),
    # LTX-2 — video generation (shallow only)
    ("ltx-video", Path.home() / "dev/LTX-2"),
]

# CLAUDE.md files from other repos (one-off)
EXTRA_FILES = [
    ("protolabs-node/CLAUDE.md", Path.home() / "dev/CLAUDE.md"),
    ("vllm-turboquant/CLAUDE.md", Path.home() / "dev/vllm-turboquant/CLAUDE.md"),
    ("protoMaker/CLAUDE.md", Path.home() / "dev/protoMaker/CLAUDE.md"),
    ("homelab-iac/CLAUDE.md", Path.home() / "dev/homelab-iac/CLAUDE.md"),
]

# File patterns to include
PATTERNS = ["*.md", "*.py", "*.sh", "*.yaml", "*.yml", "*.toml"]

# Files/dirs to skip
SKIP_PATTERNS = [
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".deps",
    "site-packages", "lib/python", "dist-info",
    "package-lock.json", "*.safetensors", "*.bin", "*.pt",
    "ComfyUI", "open-webui", "mythxengine", "rabbit-hole.io",
    "svgval", "quant-env", "tq-env", "tts-env", "vllm-env",
    "vllm-omni-env", "vllm-turboquant",
]

# Min/max file sizes
MIN_SIZE = 200  # bytes
MAX_SIZE = 50000  # bytes


def should_skip(path: Path) -> bool:
    parts = path.parts
    for skip in SKIP_PATTERNS:
        if skip.startswith("*"):
            if path.suffix == skip[1:]:
                return True
        elif skip in parts:
            return True
    return False


def extract_sections(content: str, source_id: str) -> list[dict]:
    """Split markdown/text into logical sections."""
    docs = []

    # Try splitting on markdown headers
    sections = re.split(r'\n(#{1,3}\s+.+)\n', content)

    if len(sections) > 2:
        # Has headers — group header + content pairs
        current_header = source_id
        current_content = sections[0].strip()

        for i in range(1, len(sections), 2):
            if current_content and len(current_content) > 100:
                docs.append({
                    "id": f"{source_id}::{current_header}",
                    "content": f"{current_header}\n\n{current_content}",
                })

            current_header = sections[i].strip().lstrip("#").strip()
            current_content = sections[i + 1].strip() if i + 1 < len(sections) else ""

        if current_content and len(current_content) > 100:
            docs.append({
                "id": f"{source_id}::{current_header}",
                "content": f"{current_header}\n\n{current_content}",
            })
    else:
        # No headers — treat as single doc
        if len(content.strip()) > 100:
            docs.append({"id": source_id, "content": content.strip()})

    return docs


def extract_python_docs(content: str, source_id: str) -> list[dict]:
    """Extract docstrings, comments, and function signatures from Python files."""
    docs = []

    # Extract module docstring
    match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if match:
        docs.append({
            "id": f"{source_id}::module-doc",
            "content": match.group(1).strip(),
        })

    # Extract class/function definitions with docstrings
    pattern = r'(?:class|def)\s+(\w+).*?:\s*\n\s*"""(.*?)"""'
    for match in re.finditer(pattern, content, re.DOTALL):
        name = match.group(1)
        docstring = match.group(2).strip()
        if len(docstring) > 50:
            docs.append({
                "id": f"{source_id}::{name}",
                "content": f"{name}\n\n{docstring}",
            })

    # If no docstrings found, include the whole file if it has comments
    if not docs:
        comment_lines = [l for l in content.split('\n') if l.strip().startswith('#')]
        if len(comment_lines) > 3 and len(content) > 200:
            docs.append({"id": source_id, "content": content[:MAX_SIZE]})

    return docs


def build_corpus() -> list[dict]:
    """Build corpus from all sources."""
    corpus = []
    seen_ids = set()

    for source_name, source_dir in SOURCES:
        if not source_dir.exists():
            print(f"  Skipping {source_name} — {source_dir} not found")
            continue

        for pattern in PATTERNS:
            # Limit depth for large repos
            if source_name in ("ltx-video",):
                files = list(source_dir.glob(pattern)) + list(source_dir.glob(f"*/{pattern}"))
            else:
                files = list(source_dir.rglob(pattern))

            for f in sorted(files):
                if should_skip(f):
                    continue

                try:
                    size = f.stat().st_size
                    if size < MIN_SIZE or size > MAX_SIZE:
                        continue

                    content = f.read_text(errors="ignore")
                except (OSError, UnicodeDecodeError):
                    continue

                # Build source ID from relative path
                try:
                    rel = f.relative_to(source_dir)
                except ValueError:
                    rel = f.name
                source_id = f"{source_name}/{rel}"

                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)

                # Extract documents based on file type
                if f.suffix == ".md":
                    docs = extract_sections(content, source_id)
                elif f.suffix == ".py":
                    docs = extract_python_docs(content, source_id)
                else:
                    # Shell scripts, configs — include as-is
                    if len(content.strip()) > 100:
                        docs = [{"id": source_id, "content": content.strip()[:MAX_SIZE]}]
                    else:
                        docs = []

                for doc in docs:
                    if doc["id"] not in seen_ids or doc["id"] == source_id:
                        corpus.append(doc)

    # Add extra one-off files
    for source_id, fpath in EXTRA_FILES:
        if fpath.exists() and source_id not in seen_ids:
            try:
                content = fpath.read_text(errors="ignore")
                docs = extract_sections(content, source_id)
                corpus.extend(docs)
                seen_ids.add(source_id)
            except OSError:
                pass

    return corpus


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/context-1/large_corpus.json")
    args = parser.parse_args()

    print("Building corpus from local repos...")
    corpus = build_corpus()

    # Stats
    total_chars = sum(len(d["content"]) for d in corpus)
    print(f"\nCorpus stats:")
    print(f"  Documents: {len(corpus)}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Avg doc size: {total_chars // max(len(corpus), 1):,} chars")

    # Source breakdown
    sources = {}
    for doc in corpus:
        src = doc["id"].split("/")[0]
        sources[src] = sources.get(src, 0) + 1
    print(f"\n  By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count} docs")

    output = Path(args.output)
    output.write_text(json.dumps(corpus, indent=2))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
