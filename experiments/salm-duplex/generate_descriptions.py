"""
Generate LLM descriptions from seed transcripts using local vLLM.
Runs concurrent requests for high throughput.

Usage:
    python generate_descriptions.py \
        --input /mnt/data/salm-duplex/data/train-clean-100-attributes.jsonl \
        --output /mnt/data/salm-duplex/data/train-clean-100-described.jsonl \
        --concurrency 8
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


SYSTEM_PROMPT = "You are a speech analysis assistant. Given a description of an audio clip, provide a comprehensive natural language description of what can be heard."
USER_TEMPLATE = "What can you hear from this audio?\n\n{seed}"


async def generate_one(client: httpx.AsyncClient, seed: str, model: str, url: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            resp = await client.post(
                f"{url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(seed=seed)},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 250,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR: {e}]"


async def main_async(args):
    # Load input
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    if args.max_samples:
        records = records[:args.max_samples]
        print(f"Capped to {len(records)} samples")

    semaphore = asyncio.Semaphore(args.concurrency)
    client = httpx.AsyncClient()

    start = time.time()
    tasks = []
    for r in records:
        seed = r.get("seed_transcript", "")
        tasks.append(generate_one(client, seed, args.model, args.url, semaphore))

    # Process with progress
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    with open(output_path, "w") as out_f:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            desc = await coro
            records[i]["description"] = desc  # Note: order may not match due to as_completed
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta = (len(records) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(records)}] {rate:.1f} samples/s, ETA: {eta/3600:.1f}h")

    # Re-process in order (as_completed doesn't preserve order)
    # Actually, let's use gather instead for ordered results
    await client.aclose()

    # Redo with gather for correct ordering
    client = httpx.AsyncClient()
    print(f"\nGenerating descriptions with concurrency={args.concurrency}...")
    start = time.time()

    batch_size = 500
    all_descriptions = []
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start:batch_start + batch_size]
        batch_tasks = [
            generate_one(client, r.get("seed_transcript", ""), args.model, args.url, semaphore)
            for r in batch
        ]
        batch_descs = await asyncio.gather(*batch_tasks)
        all_descriptions.extend(batch_descs)

        elapsed = time.time() - start
        done = len(all_descriptions)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(records) - done) / rate if rate > 0 else 0
        errors = sum(1 for d in all_descriptions if d.startswith("[ERROR"))
        print(f"  [{done}/{len(records)}] {rate:.1f} samples/s, ETA: {eta/3600:.1f}h, errors: {errors}")

    await client.aclose()

    # Write output
    with open(output_path, "w") as f:
        for r, desc in zip(records, all_descriptions):
            r["description"] = desc
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    errors = sum(1 for d in all_descriptions if d.startswith("[ERROR"))
    print(f"\nDone! {len(records)} samples in {elapsed:.0f}s ({len(records)/elapsed:.1f} samples/s)")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
