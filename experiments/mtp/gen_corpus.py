#!/usr/bin/env python
"""Generate a teacher corpus = the TARGET model's own generations.

We re-align a grafted MTP head to the fine-tune's hidden states, so the training
signal must come from the fine-tune itself: sample its outputs, then (in distill.py)
teach the head to predict the next-next token of those outputs from the base's hidden
states. Distilling on the model's own distribution (vs. arbitrary text) maximizes
acceptance at serve time, because that's exactly the distribution the head will draft.

Serve the target OFF-GATEWAY first (e.g. Ornith-9B on :8005, NOT :8003 which is
production), then point --url at it.

Prompts: a JSONL file, one object per line, either:
  {"messages": [{"role": "user", "content": "..."}]}   # preferred (chat)
  {"prompt": "..."}                                      # raw text -> wrapped as a user turn

Output JSONL, one object per line:
  {"messages": [...], "text": "<the model's completion>"}

Usage:
  python gen_corpus.py --url http://localhost:8005/v1 --model ornith-9b \
      --prompts prompts.jsonl --out /mnt/data/datasets/ornith-9b-mtp/corpus.jsonl \
      --n 20000 --max-tokens 1024 --temperature 0.7 --concurrency 32
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading

import requests


def load_prompts(path: str, n: int) -> list[list[dict]]:
    out: list[list[dict]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                out.append(obj["messages"])
            elif "prompt" in obj:
                out.append([{"role": "user", "content": obj["prompt"]}])
            if len(out) >= n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="OpenAI-compatible base url (off-gateway target)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL of prompts (messages/prompt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--no-think", action="store_true",
                    help="disable thinking (chat_template_kwargs.enable_thinking=false) -- avoids the "
                         "thinking-budget-exhaustion drop; keeps train/measure distributions matched")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.n)
    print(f"loaded {len(prompts)} prompts; generating with {args.model} @ {args.url}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lock = threading.Lock()
    done = [0]
    fh = open(args.out, "w")

    def work(messages: list[dict]) -> None:
        try:
            body = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if args.no_think:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            r = requests.post(
                f"{args.url}/chat/completions",
                headers={"Authorization": f"Bearer {args.api_key}"},
                json=body,
                timeout=600,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # keep going; a few failures are fine for a corpus
            text = ""
            sys.stderr.write(f"gen error: {e}\n")
        if text:
            with lock:
                fh.write(json.dumps({"messages": messages, "text": text}) + "\n")
                done[0] += 1
                if done[0] % 200 == 0:
                    fh.flush()
                    print(f"  {done[0]}/{len(prompts)}")

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, prompts))
    fh.close()
    print(f"wrote {done[0]} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
