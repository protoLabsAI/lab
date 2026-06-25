#!/bin/bash
# Drives concurrency + -O3 A/B for the dFlash smart lane on GPU1:8003 (prod untouched).
# Configs: dflash O3=1 (shipped) | dflash O3=0 (-O3 A/B) | mtp O3=1 (concurrency baseline).
set -uo pipefail
cd "$(dirname "$0")"
PORT=8003
LEVELS="1 4 8 16 32"
LOG=/mnt/scratch/logs/dflash-conc.log
RES=/mnt/scratch/logs/dflash-conc-results.txt
: > "$RES"

kill_server() {
  pkill -f "served-model-name dflash-test" 2>/dev/null || true
  for _ in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":${PORT}\b" || break; sleep 1; done
  sleep 3
}

run_cfg() {  # label MODE O3 [NUM_SPEC]
  local label="$1" mode="$2" o3="$3" ns="${4:-10}"
  echo ""                              | tee -a "$RES"
  echo "############ $label ############" | tee -a "$RES"
  : > "$LOG"
  MODE="$mode" O3="$o3" NUM_SPEC="$ns" bash run-dflash.sh > "$LOG" 2>&1 &
  local ready=0
  for _ in $(seq 1 180); do
    if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q dflash-test; then ready=1; break; fi
    if grep -qiE "Traceback|Error:|raise |No supported|out of memory|Aborted|Killed" "$LOG" 2>/dev/null; then
      echo "  $label FAILED to start (see $LOG)" | tee -a "$RES"; break; fi
    sleep 2
  done
  if [ "$ready" = 1 ]; then
    python3 conc_bench.py "$PORT" dflash-test "$LEVELS" 400 2>&1 | tee -a "$RES"
  fi
  kill_server
}

run_cfg "dFlash N=10, -O3 ON  (shipped prod config)" dflash 1 10
run_cfg "dFlash N=10, -O3 OFF (the -O3 A/B)"         dflash 0 10
run_cfg "MTP, -O3 ON          (concurrency baseline)" mtp    1 1
echo ""; echo "=== CONC DRIVER DONE ===" | tee -a "$RES"
