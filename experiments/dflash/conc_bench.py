#!/usr/bin/env python3
# Concurrency bench: for each concurrency level C, fire C simultaneous chat completions
# (fixed length via ignore_eos), measure aggregate output tok/s + mean per-request tok/s.
# Scrapes spec-decode acceptance from /metrics deltas per level.
#
# Usage: conc_bench.py PORT MODEL "1 4 8 16 32" [TOKENS]
import sys, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT  = sys.argv[1]
MODEL = sys.argv[2]
LEVELS = [int(x) for x in sys.argv[3].split()]
TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
URL = f"http://localhost:{PORT}"
PROMPT = "Write a long, detailed technical essay about distributed systems, consensus, and fault tolerance."

def metrics():
    txt = urllib.request.urlopen(f"{URL}/metrics", timeout=5).read().decode()
    m = {}
    for ln in txt.splitlines():
        if ln.startswith('#') or not ln: continue
        n = ln.split('{')[0]
        try: m[n] = m.get(n, 0) + float(ln.split(' ')[-1])
        except: pass
    return m

def one_req():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": TOKENS, "temperature": 0.7,
        "ignore_eos": True,                       # force exactly TOKENS for clean throughput
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{URL}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600).read())
    dt = time.time() - t0
    return r["usage"]["completion_tokens"], dt

# warmup
for _ in range(2): one_req()

print(f"{'conc':>4} | {'agg tok/s':>9} | {'per-req tok/s':>13} | {'p50 lat s':>9} | {'accept_len':>10} | {'rate':>5}")
print("-" * 70)
for C in LEVELS:
    b = metrics(); t0 = time.time()
    with ThreadPoolExecutor(max_workers=C) as ex:
        res = list(ex.map(lambda _: one_req(), range(C)))
    wall = time.time() - t0; a = metrics()
    toks = sum(x[0] for x in res)
    lats = sorted(x[1] for x in res)
    per_req = sum(x[0] / x[1] for x in res) / C
    p50 = lats[len(lats)//2]
    g = lambda k: a.get(k, 0) - b.get(k, 0)
    nd = g('vllm:spec_decode_num_drafts_total'); acc = g('vllm:spec_decode_num_accepted_tokens_total'); drf = g('vllm:spec_decode_num_draft_tokens_total')
    al = acc/nd if nd else 0; rate = 100*acc/drf if drf else 0
    print(f"{C:>4} | {toks/wall:>9.1f} | {per_req:>13.1f} | {p50:>9.2f} | {al:>10.2f} | {rate:>4.0f}%", flush=True)
