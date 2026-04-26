#!/usr/bin/env python3
"""Generate synthetic ORBIS-style fact-recall benchmark dataset.

Creates a set of facts (things a user might tell a companion AI) and
queries (how they'd ask about those facts later). Each query maps to
1-3 gold facts. Also generates distractor facts for retrieval noise.

Output: data/facts.jsonl + data/queries.jsonl

Usage:
    python scripts/generate_facts.py                    # 100 facts + 100 queries
    python scripts/generate_facts.py --n-facts 50       # smaller for testing
    python scripts/generate_facts.py --base-url http://ava:4000/v1
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests

SYSTEM_PROMPT_FACTS = """You are generating synthetic data for a voice companion AI called ORBIS.
Generate realistic personal facts that a user might share with their AI companion over weeks/months.
Facts should be diverse: family, work, hobbies, preferences, health, pets, routines, etc.
Each fact should be a single sentence, natural and conversational.
Output one fact per line, no numbering."""

SYSTEM_PROMPT_QUERIES = """You are generating test queries for a memory retrieval system.
Given a list of facts a user previously shared with their AI companion ORBIS,
generate natural voice queries the user might ask to recall those facts.
Queries should sound like spoken language (as from speech transcription).
Include indirect references ("that thing I mentioned about..."), fuzzy recalls
("something about my sister?"), and direct recalls ("what's my cat's name?").

For each query, also output which fact number(s) it refers to.

Format each line as: QUERY ||| FACT_NUMBERS
Example: what was my cat's name again ||| 3
Example: tell me about my family ||| 1,5,12"""


def generate_facts(n: int, base_url: str, model: str) -> list[str]:
    """Generate n personal facts via LLM."""
    all_facts = []
    batch_size = 50

    while len(all_facts) < n:
        remaining = n - len(all_facts)
        batch_n = min(batch_size, remaining)

        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_FACTS},
                    {"role": "user", "content": f"Generate {batch_n} diverse personal facts."},
                ],
                "temperature": 0.9,
                "max_tokens": 2048,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            timeout=120,
        )
        resp.raise_for_status()

        msg = resp.json()["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        lines = [
            line.strip().lstrip("0123456789.-) •*").strip().strip('"\'')
            for line in text.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        all_facts.extend(lines)
        time.sleep(0.5)

    return all_facts[:n]


def generate_queries(facts: list[str], n: int, base_url: str, model: str) -> list[dict]:
    """Generate n queries with gold fact mappings.

    Presents 10 facts at a time to keep context short and generation fast.
    Each batch generates 5 queries referencing those 10 facts.
    """
    all_queries = []
    fact_window = 10
    queries_per_window = 5

    indices = list(range(len(facts)))

    while len(all_queries) < n:
        # Pick a random window of facts
        random.shuffle(indices)
        window_ids = sorted(indices[:fact_window])
        facts_block = "\n".join(f"{i}: {facts[i]}" for i in window_ids)

        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_QUERIES},
                    {
                        "role": "user",
                        "content": f"Here are the facts:\n\n{facts_block}\n\nGenerate {queries_per_window} queries. Format: QUERY ||| FACT_NUMBERS",
                    },
                ],
                "temperature": 0.8,
                "max_tokens": 1024,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            timeout=120,
        )
        resp.raise_for_status()

        msg = resp.json()["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        for line in text.strip().split("\n"):
            if "|||" not in line:
                continue
            parts = line.split("|||")
            if len(parts) != 2:
                continue
            query = parts[0].strip().lstrip("0123456789.-) •*").strip().strip('"\'')
            try:
                fact_ids = [int(x.strip()) for x in parts[1].strip().split(",")]
                # Validate indices are in our window
                fact_ids = [i for i in fact_ids if i in window_ids]
                if query and fact_ids:
                    all_queries.append({
                        "query": query,
                        "gold_fact_ids": fact_ids,
                        "gold_facts": [facts[i] for i in fact_ids],
                    })
            except ValueError:
                continue

        time.sleep(0.5)
        print(f"    {len(all_queries)}/{n} queries...", flush=True)

    return all_queries[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fact-recall dataset")
    parser.add_argument("--n-facts", type=int, default=100)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="local")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent.parent / "data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check LLM
    try:
        r = requests.get(f"{args.base_url}/models", timeout=5)
        print(f"LLM: {args.base_url}")
    except requests.ConnectionError:
        print(f"ERROR: LLM not reachable at {args.base_url}")
        return

    # Generate facts
    print(f"\nGenerating {args.n_facts} facts...")
    facts = generate_facts(args.n_facts, args.base_url, args.model)
    print(f"  Got {len(facts)} facts")

    # Generate queries
    print(f"\nGenerating {args.n_queries} queries...")
    queries = generate_queries(facts, args.n_queries, args.base_url, args.model)
    print(f"  Got {len(queries)} queries")

    # Write outputs
    facts_path = output_dir / "facts.jsonl"
    with open(facts_path, "w") as f:
        for i, fact in enumerate(facts):
            f.write(json.dumps({"id": i, "text": fact}) + "\n")

    queries_path = output_dir / "queries.jsonl"
    with open(queries_path, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    print(f"\nFacts:   {facts_path} ({len(facts)} items)")
    print(f"Queries: {queries_path} ({len(queries)} items)")

    # Print samples
    print("\n=== Sample facts ===")
    for fact in random.sample(facts, min(5, len(facts))):
        print(f"  {fact}")

    print("\n=== Sample queries ===")
    for q in random.sample(queries, min(5, len(queries))):
        print(f"  Q: {q['query']}")
        print(f"  Gold: {q['gold_facts']}")


if __name__ == "__main__":
    main()
