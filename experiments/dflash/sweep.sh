#!/bin/bash
# Sweep num_speculative_tokens for the dFlash smart-lane draft.
# For each N: serve on GPU1:8003, wait ready, bench (decode tok/s + acceptance), tear down.
# Emits one result line per N. Prod lanes untouched.
#
# Usage: bash sweep.sh "3 6 10 16"   (default list below)
set -uo pipefail
cd "$(dirname "$0")"
LAB="$(cd ../.. && pwd)"
NS_LIST="${1:-3 6 10 16}"
PORT=8003
LOG=/mnt/scratch/logs/dflash-sweep.log
RESULTS=/mnt/scratch/logs/dflash-sweep-results.txt
: > "$RESULTS"

kill_server() {
  pkill -f "served-model-name dflash-test" 2>/dev/null || true
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":${PORT}\b" || break; sleep 1; done
  sleep 2
}

bench_one() {  # returns "decode_tok/s | accept_len | rate%"
  python3 - "$PORT" <<'PY'
import sys,urllib.request,json,time
port=sys.argv[1]
def metrics():
    txt=urllib.request.urlopen(f"http://localhost:{port}/metrics",timeout=5).read().decode()
    m={}
    for ln in txt.splitlines():
        if ln.startswith('#') or not ln: continue
        n=ln.split('{')[0]
        try: m[n]=m.get(n,0)+float(ln.split(' ')[-1])
        except: pass
    return m
prompt="Write a detailed 500-word essay on the history of computing from Babbage to modern AI."
body=json.dumps({"model":"dflash-test","messages":[{"role":"user","content":prompt}],
   "max_tokens":800,"temperature":0.7}).encode()  # thinking ON (smart lane is a thinking lane)
def post():
    return json.loads(urllib.request.urlopen(urllib.request.Request(
      f"http://localhost:{port}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=180).read())
for _ in range(2): post()              # warmup
b=metrics(); t0=time.time(); gen=0
# accumulate TPOT-based decode rate from /metrics deltas
for _ in range(6): gen+=post()["usage"]["completion_tokens"]
dt=time.time()-t0; a=metrics()
g=lambda k:a.get(k,0)-b.get(k,0)
tpot_sum=g('vllm:request_time_per_output_token_seconds_sum'); tpot_cnt=g('vllm:request_time_per_output_token_seconds_count')
decode = (tpot_cnt/tpot_sum) if tpot_sum else 0
acc=g('vllm:spec_decode_num_accepted_tokens_total'); drf=g('vllm:spec_decode_num_draft_tokens_total'); nd=g('vllm:spec_decode_num_drafts_total')
print(f"{decode:6.1f} | wall {gen/dt:6.1f} | accept_len {acc/nd if nd else 0:.2f} | rate {100*acc/drf if drf else 0:4.1f}%")
PY
}

for N in $NS_LIST; do
  echo "### N=$N : starting server…" | tee -a "$RESULTS"
  : > "$LOG"
  NUM_SPEC=$N MODE=dflash bash run-dflash.sh > "$LOG" 2>&1 &
  # wait ready (up to ~5 min)
  ready=0
  for _ in $(seq 1 150); do
    if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q dflash-test; then ready=1; break; fi
    if grep -qiE "Traceback|Error:|raise |No supported|out of memory|Aborted" "$LOG" 2>/dev/null; then echo "  N=$N FAILED to start (see $LOG)" | tee -a "$RESULTS"; break; fi
    sleep 2
  done
  if [ "$ready" = 1 ]; then
    line=$(bench_one)
    echo "N=$N : decode $line" | tee -a "$RESULTS"
  fi
  kill_server
done
echo "=== SWEEP DONE ===" | tee -a "$RESULTS"
