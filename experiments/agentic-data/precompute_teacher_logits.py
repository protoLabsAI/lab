"""Precompute teacher (Ornith-35B) top-k logprobs for white-box KD, off the SERVED :8000.

For each trajectory we reproduce train_lora.build_example's exact templated token ids (shared
248K vocab → ids are identical teacher↔student), send them to the vLLM completions API with
`echo=true, logprobs=k` (a single prefill, no generation), and keep the top-k logprobs ONLY at
the assistant-masked positions (labels != -100 — the only positions the KD loss touches).

Cache shape (npz per shard): for each kept position i (in the student's token sequence):
  pos[i]        int  — index into input_ids
  topk_ids[i]   int[k]
  topk_lp[i]    f16[k]
aligned so teacher position t is the distribution predicting input_ids[t] (the −1 shift is
applied at loss time, matching CE label shift).

  python precompute_teacher_logits.py --data <jsonl> --out <dir> --k 16 [--limit N]
"""
from __future__ import annotations
import argparse, json, os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")
import numpy as np, urllib.request
from transformers import AutoTokenizer
from train_lora import build_example, _flatten

TEACHER_URL = "http://localhost:8020/v1/completions"  # eager Ornith-35B w/ --return-tokens-as-token-ids


def teacher_topk(token_ids: list[int], k: int, model="teacher") -> list[dict]:
    """Per-position top-k as {int token_id: logprob} via echo. The server runs
    --return-tokens-as-token-ids so top_logprobs keys are 'token_id:N' — parse to int."""
    payload = {"model": model, "prompt": token_ids, "max_tokens": 1,
               "echo": True, "logprobs": k, "temperature": 0}
    req = urllib.request.Request(TEACHER_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    tl = d["choices"][0]["logprobs"].get("top_logprobs") or []
    out = []
    for pos in tl:
        if not pos:
            out.append(None); continue
        out.append({int(kk.split(":", 1)[1]): float(v) for kk, v in pos.items()
                    if kk.startswith("token_id:")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    id_of = tok.convert_tokens_to_ids

    rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        rows = rows[:a.limit]
    kept = 0
    for idx, r in enumerate(rows):
        ex = build_example(_flatten(r["messages"]), tok, a.max_seq)
        if not ex:
            continue
        ids, labels = ex["input_ids"], ex["labels"]
        asst_pos = [i for i, l in enumerate(labels) if l != -100]
        if not asst_pos:
            continue
        try:
            top = teacher_topk(ids, a.k)
        except Exception as e:
            print(f"[warn] traj {idx}: {e}")
            continue
        P, TID, TLP = [], [], []
        for i in asst_pos:
            if i >= len(top) or not top[i]:
                continue
            entries = sorted(top[i].items(), key=lambda e: -e[1])[:a.k]  # {int id: logprob}
            tids = [e[0] for e in entries]; tlps = [e[1] for e in entries]
            while len(tids) < a.k:
                tids.append(-1); tlps.append(-1e4)
            P.append(i); TID.append(tids); TLP.append(tlps)
        if not P:
            continue
        np.savez_compressed(os.path.join(a.out, f"{idx:06d}.npz"),
                            pos=np.array(P, np.int32),
                            topk_ids=np.array(TID, np.int32),
                            topk_lp=np.array(TLP, np.float16))
        kept += 1
        if kept % 100 == 0:
            print(f"  cached {kept} trajectories (idx {idx})")
    print(f"DONE: {kept} trajectory logit-caches -> {a.out}")


if __name__ == "__main__":
    main()
