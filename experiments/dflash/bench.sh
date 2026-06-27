#!/bin/bash
# A/B bench harness for the dflash experiment.
# Runs models/speed-test.sh against a given port, then scrapes spec-decode acceptance
# metrics from /metrics (acceptance length is the headline number for spec decode).
#
# Usage: bash bench.sh [PORT] [RUNS] [MODE]
set -euo pipefail
PORT="${1:-8003}"
RUNS="${2:-5}"
MODE="${3:-long}"
URL="http://localhost:${PORT}"
LAB="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== dflash bench :${PORT} ==="
bash "${LAB}/models/speed-test.sh" "$RUNS" "$MODE" "$URL" || true

echo
echo "=== spec-decode acceptance (from /metrics) ==="
curl -s "${URL}/metrics" 2>/dev/null | python3 -c "
import sys
acc_sum=acc_cnt=draft=accepted=0.0
per_pos={}
for ln in sys.stdin:
    ln=ln.strip()
    if ln.startswith('#') or not ln: continue
    name=ln.split('{')[0]; val=float(ln.split(' ')[-1])
    if name=='vllm:spec_decode_num_accepted_tokens_total': accepted=val
    elif name=='vllm:spec_decode_num_draft_tokens_total': draft=val
    elif name=='vllm:spec_decode_num_drafts_total': acc_cnt=val
    elif name.startswith('vllm:spec_decode_num_accepted_tokens_per_pos'):
        per_pos[ln.split('position=\"')[-1].split('\"')[0] if 'position' in ln else '?']=val
if acc_cnt:
    print(f'mean accepted len/draft : {accepted/acc_cnt:.2f} tokens')
if draft:
    print(f'draft acceptance rate   : {100*accepted/draft:.1f}%  ({int(accepted)}/{int(draft)})')
if not acc_cnt and not draft:
    print('(no spec_decode metrics — baseline/no-spec run, or metric names differ on this build)')
"
