"""Prose → structured tag extractor.

Reads DeSTA2 description text from
`/mnt/data/salm-duplex/manifests/full-960h-described/{train,val}.jsonl.gz`
and asks the local Qwen vLLM to extract gender / mood / environment /
speech_style tags. Per-tag confidence reported as 'high'/'medium'/'low'
based on whether the description states the attribute explicitly.

Output: jsonl with keys {audio_path, tags{...}, raw_text} per line.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taxonomy import HEADS_BY_NAME  # noqa: E402

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "local"

PROSE_HEADS = ("speaker_gender", "speaker_age", "mood_class",
               "environment", "speech_style")


def _build_prompt(desc: str) -> list[dict]:
    classes = {h: HEADS_BY_NAME[h].classes for h in PROSE_HEADS}
    schema_lines = "\n".join(
        f"  - {h}: one of {list(classes[h])}" for h in PROSE_HEADS
    )
    sys_msg = (
        "You extract structured audio tags from a description of an audio clip. "
        "Output ONLY valid JSON. No prose. If a tag is not stated in the "
        "description, use 'unknown'. Use 'medium'/'low' confidence when the "
        "tag is implied but not stated; 'high' when stated explicitly.\n\n"
        f"Schema:\n{schema_lines}\n\n"
        "For every tag also emit a confidence: high|medium|low.\n\n"
        'Output shape: {"speaker_gender": {"value": "...", "confidence": "..."}, ...}'
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Description:\n{desc}\n\nReturn JSON only."},
    ]


async def _extract_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, desc: str
) -> dict | None:
    async with sem:
        try:
            r = await client.post(
                VLLM_URL,
                json={
                    "model": MODEL,
                    "messages": _build_prompt(desc),
                    "temperature": 0.0,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            return {"_error": str(e)[:200]}


def _iter_descriptions(path: Path, limit: int | None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            d = json.loads(line)
            sup = d.get("supervisions") or []
            if not sup:
                continue
            text = sup[0]["text"]
            audio_path = (d.get("recording") or {}).get("sources", [{}])[0].get("source")
            yield {"audio_path": audio_path, "text": text}


async def _run(input_path: Path, output_path: Path, limit: int | None, concurrency: int):
    items = list(_iter_descriptions(input_path, limit))
    print(f"Loaded {len(items)} descriptions from {input_path.name}", flush=True)

    done_paths: set[str] = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    done_paths.add(json.loads(line)["audio_path"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(done_paths):,} already done", flush=True)

    pending = [it for it in items if it["audio_path"] not in done_paths]
    if not pending:
        print("Nothing to do.", flush=True)
        return
    print(f"Pending: {len(pending):,}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _process(client, item):
        tags = await _extract_one(client, sem, item["text"])
        return {"audio_path": item["audio_path"], "text": item["text"], "tags": tags}

    n_ok = n_err = 0
    t0 = time.time()
    last_log = t0
    out_f = output_path.open("a")
    try:
        async with httpx.AsyncClient() as client:
            coros = [_process(client, it) for it in pending]
            for fut in asyncio.as_completed(coros):
                res = await fut
                out_f.write(json.dumps(res) + "\n")
                tags = res.get("tags") or {}
                if "_error" in tags:
                    n_err += 1
                else:
                    n_ok += 1
                now = time.time()
                if now - last_log >= 30:
                    done = n_ok + n_err
                    rate = done / max(now - t0, 1e-3)
                    eta = (len(pending) - done) / max(rate, 1e-3)
                    print(f"  [{done:,}/{len(pending):,}]  ok={n_ok:,} err={n_err:,}  "
                          f"{rate:.1f} req/s  eta {eta/3600:.2f} h",
                          flush=True)
                    out_f.flush()
                    last_log = now
    finally:
        out_f.close()
    elapsed = time.time() - t0
    print(f"Done. ok={n_ok:,} err={n_err:,}  elapsed {elapsed/3600:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/data/salm-duplex/manifests/full-960h-described/train.jsonl.gz")
    ap.add_argument("--output", default="/mnt/data/audio-tags/labels/prose-pilot.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    asyncio.run(_run(Path(args.input), Path(args.output), args.limit, args.concurrency))


if __name__ == "__main__":
    main()
