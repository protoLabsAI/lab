#!/usr/bin/env bash
# Rung-1 generation queue: after the running telecom pass finishes, roll straight through
# retail + airline (single greedy pass each) against Ornith :8000, filter each to canonical,
# then report the combined verified corpus size. Fires GATE-1-READY when done.
#
# Robust-by-design: waits on the telecom PID via `kill -0` (NOT pgrep pattern-match — avoids
# the self-match trap). Each subsequent domain runs in the FOREGROUND of this driver, so the
# driver naturally blocks until it finishes; we capture each child's own PID for logging only.
set -u
cd /home/ava/dev/lab/experiments/agentic-data/tau2-bench
RAW=/mnt/data/datasets/agentic-distill/_raw
FILT=/home/ava/dev/lab/experiments/agentic-data/filter_tau2.py
LOG=/mnt/scratch/gen-queue.log
TELECOM_PID=2462794
ARGS='{"api_base":"http://localhost:8000/v1","api_key":"dummy"}'
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "gen_queue start. waiting on telecom pid $TELECOM_PID ..."
while kill -0 "$TELECOM_PID" 2>/dev/null; do sleep 60; done
say "telecom pass finished. filtering ..."
python3 "$FILT" "$RAW/tau2-telecom" telecom --append 2>&1 | tee -a "$LOG"

run_domain(){
  local dom="$1" conc="$2"
  say "=== generating $dom (1 greedy pass, conc $conc) ==="
  .venv/bin/tau2 run --domain "$dom" \
    --agent-llm openai/local --agent-llm-args "$ARGS" \
    --user-llm  openai/local --user-llm-args  "$ARGS" \
    --task-split-name full --max-concurrency "$conc" \
    --save-to "$RAW/tau2-$dom" >>"$LOG" 2>&1
  say "$dom done. filtering ..."
  python3 "$FILT" "$RAW/tau2-$dom" "$dom" --append 2>&1 | tee -a "$LOG"
}

run_domain retail 8
run_domain airline 8

# Combined verified corpus: tonight's v1 (ornith_tau.jsonl) + all tau2 (ornith_tau2.jsonl),
# deduped by full-content hash (env-traj sets share boilerplate openers → hash whole sequence).
say "=== building combined corpus (dedup) ==="
python3 - "$RAW" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, hashlib
raw=sys.argv[1]
seen=set(); rows=[]; per={}
for name in ("ornith_tau.jsonl","ornith_tau2.jsonl"):
    p=f"{raw}/{name}"
    try: f=open(p)
    except FileNotFoundError: continue
    for line in f:
        r=json.loads(line)
        h=hashlib.sha1(json.dumps(r.get("messages"),sort_keys=True).encode()).hexdigest()
        if h in seen: continue
        seen.add(h); rows.append(r)
        per[r.get("domain","?")]=per.get(r.get("domain","?"),0)+1
out="/mnt/data/datasets/agentic-distill/corpus_rung1.jsonl"
with open(out,"w") as fo:
    for r in rows: fo.write(json.dumps(r)+"\n")
print(f"  combined unique verified: {len(rows)}  -> {out}")
for d,c in sorted(per.items(),key=lambda x:-x[1]): print(f"    {d}: {c}")
PY
say "=== GATE-1-READY === rung-1 generation complete."
