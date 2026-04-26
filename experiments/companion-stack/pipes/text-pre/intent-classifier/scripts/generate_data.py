#!/usr/bin/env python3
"""Generate synthetic intent classification training data.

Uses the local LLM (vLLM on :8000 or gateway on ava:4000) to generate
realistic ORBIS user utterances for each intent class.

Output: data/intent_train.jsonl + data/intent_test.jsonl

Usage:
    python scripts/generate_data.py                    # 300/class via localhost:8000
    python scripts/generate_data.py --n-per-class 100  # smaller for testing
    python scripts/generate_data.py --base-url http://ava:4000/v1  # via gateway
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests

CLASSES = {
    "chat": {
        "description": "Casual conversation, emotional sharing, jokes, small talk, opinions. No tools needed.",
        "examples": [
            "I'm feeling tired today",
            "Tell me a joke",
            "What do you think about that?",
            "How was your day?",
            "I love this song",
            "Do you ever get lonely?",
        ],
    },
    "command": {
        "description": "Direct orb commands: change visual appearance, palette, warmth, personality, save/recall presets. Maps to ORBIS tools: set_variant, apply_palette, adjust_param, save_preset, recall_preset, adjust_personality.",
        "examples": [
            "Be warmer",
            "Set the palette to ocean",
            "Make yourself more playful",
            "Save this as my cozy preset",
            "Switch to the aurora variant",
            "Turn down the brightness",
        ],
    },
    "delegate": {
        "description": "Tasks that need delegation to external agents: coding help, web search, research, complex tasks the orb can't handle directly. Maps to delegate_to() tool.",
        "examples": [
            "Can you help me debug this Python error?",
            "Search for the latest news about SpaceX",
            "Write me a function that sorts a list",
            "Research the best restaurants nearby",
            "Help me draft an email to my boss",
            "Look up how to fix a leaky faucet",
        ],
    },
    "memory": {
        "description": "Queries about past conversations, stored facts, things the user previously told the orb. Requires retrieval from memory before LLM response.",
        "examples": [
            "Remember when I told you about my sister?",
            "What did we talk about last time?",
            "Do you remember my favorite color?",
            "What was that recipe I mentioned?",
            "Didn't I tell you about my trip?",
            "What do you know about me?",
        ],
    },
    "meta": {
        "description": "Ambiguous, multi-intent, greetings, meta-conversation about the orb itself, requests that don't clearly fit another category. This is the fallback/catch-all.",
        "examples": [
            "Hey",
            "What can you do?",
            "That's interesting, and also can you search for...",
            "Hmm",
            "Never mind",
            "What are you?",
        ],
    },
}

SYSTEM_PROMPT = """You are a data generator for an intent classification system.
You will generate realistic user utterances for a voice assistant called ORBIS.
ORBIS is a companion AI orb — users talk to it naturally via voice.

Important:
- Generate utterances as they would be SPOKEN (transcribed from speech), not typed.
- Include natural speech patterns: filler words, incomplete sentences, casual grammar.
- Vary length: some short (2-3 words), some medium (5-10 words), some longer.
- Include edge cases: mumbled, ambiguous, or borderline examples.
- Do NOT include the class label in the utterance.
- Each utterance should be on its own line, no numbering or bullets.
- Generate EXACTLY {n} utterances, no more, no less."""

USER_PROMPT = """Generate {n} realistic spoken utterances for the intent class "{cls}".

Class description: {description}

Example utterances for reference (generate NEW ones, don't repeat these):
{examples}

Output {n} utterances, one per line. No numbering, no bullets, no labels."""


def generate_for_class(
    cls: str,
    info: dict,
    n: int,
    base_url: str,
    model: str,
) -> list[str]:
    """Generate n utterances for a single class via the LLM."""
    # Generate in batches to improve diversity
    batch_size = min(50, n)
    all_utterances = []

    while len(all_utterances) < n:
        remaining = n - len(all_utterances)
        batch_n = min(batch_size, remaining)

        examples_str = "\n".join(f"- {ex}" for ex in info["examples"])

        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(n=batch_n)},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(
                            n=batch_n,
                            cls=cls,
                            description=info["description"],
                            examples=examples_str,
                        ),
                    },
                ],
                "temperature": 0.9,
                "max_tokens": 2048,
                # Disable thinking mode for data generation (Qwen3.x)
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            timeout=120,
        )
        resp.raise_for_status()

        msg = resp.json()["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""

        # Parse lines, filter empty/numbered
        lines = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Strip numbering prefixes
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.-) ").strip()
            # Strip bullet prefixes
            line = line.lstrip("•-* ").strip()
            # Strip quotes
            line = line.strip('"\'')
            if line and len(line) > 1:
                lines.append(line)

        all_utterances.extend(lines)
        time.sleep(0.5)  # brief pause between batches

    return all_utterances[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic intent data")
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="local")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent.parent / "data"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check LLM is reachable
    try:
        r = requests.get(f"{args.base_url}/models", timeout=5)
        models = r.json()
        print(f"LLM: {args.base_url}")
        if "data" in models:
            print(f"Model: {models['data'][0]['id']}")
    except requests.ConnectionError:
        print(f"ERROR: LLM not reachable at {args.base_url}")
        return

    # Generate data for each class
    all_samples = []
    for cls, info in CLASSES.items():
        print(f"\nGenerating {args.n_per_class} samples for '{cls}'...")
        utterances = generate_for_class(
            cls, info, args.n_per_class, args.base_url, args.model
        )
        for utt in utterances:
            all_samples.append({"text": utt, "label": cls})
        print(f"  Got {len(utterances)} samples")

    # Shuffle and split 80/20
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train = all_samples[:split_idx]
    test = all_samples[split_idx:]

    # Write JSONL
    train_path = output_dir / "intent_train.jsonl"
    test_path = output_dir / "intent_test.jsonl"

    for path, data in [(train_path, train), (test_path, test)]:
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        print(f"\nWrote {len(data)} samples to {path}")

    # Print distribution
    print("\n=== Distribution ===")
    for split_name, data in [("train", train), ("test", test)]:
        counts = {}
        for s in data:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        print(f"\n{split_name}:")
        for cls, count in sorted(counts.items()):
            print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
