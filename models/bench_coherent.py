#!/usr/bin/env python3
"""Coherent-prompt serving benchmark — the gap speed-test-v2 leaves open.

speed-test-v2 uses `--dataset-name random`, which is fine for most lanes but
SILENTLY DEFEATS SPECULATIVE DECODING: a draft head cannot predict random
tokens. Measured on the DSV4/jasl lane 2026-08-11, DSpark acceptance was
11-22% on random data vs 39-48% on real traffic. Any spec-decode lane
benchmarked on random data is measured with its headline feature disabled.

This sends real English prose at controlled concurrency and reports the same
shape of numbers (TTFT / TPOT percentiles, aggregate throughput), so a
spec-decode lane can be compared against itself with the feature on and off.

Usage:
  python bench_coherent.py --url http://localhost:8041/v1 --model smart \
      --concurrency 8 --n 32 --input-tokens 1024 --output-tokens 256

Honest-numbers notes (see feedback_speed_numbers_honest):
  * Single-stream (C=1) numbers are NOT publishable on their own — sweep it.
  * Prompts are drawn from a shared corpus, so successive requests may share a
    prefix; pass --unique-prefix to defeat prefix caching (conservative), or
    leave it off to measure the cache-warm path prod actually gets.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_CORPUS = "/home/ava/dev/vllm-jasl-src/benchmarks/sonnet.txt"


def build_prompt(corpus: list[str], approx_tokens: int, rng: random.Random,
                 unique_prefix: bool) -> str:
    # ~0.75 words/token for English prose is close enough for prompt sizing;
    # exact length does not matter as long as both arms see the same corpus.
    words_needed = int(approx_tokens * 0.75)
    start = rng.randrange(0, max(1, len(corpus) - 1))
    out: list[str] = []
    i = start
    while sum(len(s.split()) for s in out) < words_needed:
        out.append(corpus[i % len(corpus)])
        i += 1
    body = " ".join(out)
    if unique_prefix:
        body = f"[req {rng.random():.12f}] " + body
    return ("Continue this passage in the same voice, at length.\n\n" + body)


def one_request(url: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    # Count any generated token, reasoning included — on this lane
                    # thinking tokens are real decode work and belong in TPOT.
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
    except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:120]}
    total = time.perf_counter() - t0
    if ttft is None:
        return {"ok": False, "err": "no content tokens"}
    out_tok = (usage or {}).get("completion_tokens") or chunks
    tpot = (total - ttft) / max(1, out_tok - 1)
    return {"ok": True, "ttft": ttft, "total": total, "out_tok": out_tok,
            "tpot": tpot}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8041/v1")
    ap.add_argument("--model", default="smart")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unique-prefix", action="store_true",
                    help="Defeat prefix caching (conservative, cache-COLD).")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as fh:
        corpus = [ln.strip() for ln in fh if ln.strip()]
    if not corpus:
        print("empty corpus", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    prompts = [build_prompt(corpus, args.input_tokens, rng, args.unique_prefix)
               for _ in range(args.n)]

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(
            lambda p: one_request(args.url, args.model, p,
                                  args.output_tokens, args.timeout),
            prompts))
    wall = time.perf_counter() - wall0

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    if not ok:
        print(f"ALL {len(results)} REQUESTS FAILED; first: "
              f"{bad[0].get('err') if bad else '?'}", file=sys.stderr)
        return 1

    ttfts = [r["ttft"] * 1000 for r in ok]
    tpots = [r["tpot"] * 1000 for r in ok]
    tot_out = sum(r["out_tok"] for r in ok)
    row = {
        "label": args.label, "concurrency": args.concurrency,
        "n": args.n, "ok": len(ok), "failed": len(bad),
        "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
        "ttft_p50_ms": pct(ttfts, 50), "ttft_p99_ms": pct(ttfts, 99),
        "tpot_p50_ms": pct(tpots, 50), "tpot_p99_ms": pct(tpots, 99),
        "mean_ttft_ms": statistics.fmean(ttfts),
        "mean_tpot_ms": statistics.fmean(tpots),
        "agg_out_tok_s": tot_out / wall, "wall_s": wall,
        "total_output_tokens": tot_out,
    }
    print(f"{args.label:<14} C={args.concurrency:<3} "
          f"ttft p50={row['ttft_p50_ms']:>8.0f}ms p99={row['ttft_p99_ms']:>8.0f}ms  "
          f"tpot p50={row['tpot_p50_ms']:>6.1f}ms  "
          f"agg={row['agg_out_tok_s']:>7.1f} tok/s  ok={len(ok)}/{args.n}"
          + (f"  FAILED={len(bad)}" if bad else ""))
    if bad:
        print(f"  first failure: {bad[0].get('err')}", file=sys.stderr)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
