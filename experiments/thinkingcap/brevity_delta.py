"""Measure thinking-token brevity: ThinkingCap-NVFP4 vs base Qwen3.6-27B on the same
reasoning prompts, same sampling (temp 0.6). Metric = per-prompt (base-cap)/base token
saving, averaged over prompts both answer. Run once per endpoint, then compare.

  python brevity_delta.py <base_url> <model> <label>       # e.g. http://127.0.0.1:8011/v1 thinkingcap cap
  python brevity_delta.py <compare> a.json b.json           # print the delta table
"""
import json, sys, time, urllib.request

PROMPTS = [
    "A store had 120 apples. They sold 45 in the morning and 30 in the afternoon, then received a delivery of 60 more. How many apples now?",
    "If a train travels 240 miles in 3 hours, then slows to 50 mph for the next 2 hours, what is its average speed over the whole trip?",
    "A rectangle's length is 3 more than twice its width. If the perimeter is 36, what are the dimensions?",
    "What is the sum of all integers from 1 to 100 that are divisible by 3 or 5?",
    "A shirt is discounted 20%, then another 15% off the reduced price. What single discount is that equivalent to?",
    "Three friends split a bill. Alice pays twice what Bob pays; Carol pays $6 more than Bob. If the total is $54, how much does each pay?",
    "How many trailing zeros are in 25 factorial?",
    "A tank fills in 6 hours with pipe A and 4 hours with pipe B. How long with both open?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "A number is 4 times its reciprocal plus 3. Find all such positive numbers.",
    "In a class of 30, 18 play soccer, 15 play tennis, and 5 play neither. How many play both?",
    "What is the next number in the sequence 2, 6, 12, 20, 30, ...?",
    "A ladder leans against a wall reaching 12 feet high with its base 5 feet out. How long is the ladder?",
    "If today is Wednesday, what day is it 100 days from now?",
    "A jar has red and blue marbles in ratio 3:5. After adding 8 red, the ratio is 1:1. How many blue marbles?",
]


def run(base_url, model, label):
    out = []
    for i, p in enumerate(PROMPTS):
        payload = {"model": model, "messages": [{"role": "user", "content": p}],
                   "max_tokens": 8000, "temperature": 0.6, "top_p": 0.95}
        t0 = time.time()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            m = d["choices"][0]["message"]
            ct = d["usage"]["completion_tokens"]
            ans = (m.get("content") or "") + (m.get("reasoning_content") or "")
            fin = d["choices"][0]["finish_reason"]
        except Exception as e:
            ct, ans, fin = -1, f"ERR {e}", "error"
        out.append({"i": i, "tokens": ct, "finish": fin, "answer_tail": ans[-120:]})
        print(f"  [{label}] {i:2d}: {ct:5d} tok ({fin}) {time.time()-t0:.0f}s")
    path = f"/mnt/scratch/brevity_{label}.json"
    json.dump(out, open(path, "w"))
    toks = [r["tokens"] for r in out if r["tokens"] > 0]
    print(f"[{label}] mean completion tokens: {sum(toks)/len(toks):.0f}  (n={len(toks)}) -> {path}")


def compare(a, b):
    A = {r["i"]: r for r in json.load(open(a))}
    B = {r["i"]: r for r in json.load(open(b))}
    print(f"{'#':>2} {'base':>7} {'cap':>7} {'saving':>8}")
    savings = []
    for i in sorted(A):
        ta, tb = A[i]["tokens"], B[i]["tokens"]
        if ta > 0 and tb > 0:
            s = (ta - tb) / ta
            savings.append(s)
            print(f"{i:>2} {ta:>7} {tb:>7} {s*100:>7.1f}%")
    print(f"\nMEAN per-prompt token saving: {sum(savings)/len(savings)*100:.1f}%  (n={len(savings)})")
    print(f"base mean {sum(A[i]['tokens'] for i in A if A[i]['tokens']>0)/len(A):.0f} tok  ->  "
          f"cap mean {sum(B[i]['tokens'] for i in B if B[i]['tokens']>0)/len(B):.0f} tok")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
