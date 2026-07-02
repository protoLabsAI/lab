#!/bin/bash
cd /home/ava/dev/lab/experiments/agentworld/scaffold
set -a; source /home/ava/dev/lab/evals/.env; set +a
source /home/ava/dev/lab/.venv/bin/activate
TBASE=../../../evals/claw/claw-eval/tasks
declare -A PORT=( [T104_packet_decoder]=500 [T102_xss_filter]=700 [T100_reverse_decoder]=900 )
for T in T104_packet_decoder T102_xss_filter T100_reverse_decoder; do
  echo "###### $T @ $(date +%H:%M:%S) ######"
  echo "## sim scaffold"
  python sim_practice.py --task-dir $TBASE/$T \
    --policy-endpoint http://ava:4000/v1 --policy-model protolabs/smart --policy-key "$GATEWAY_API_KEY" \
    --aw-endpoint http://localhost:8010/v1 --episodes 1 --max-turns 12 > logs/sim_$T.log 2>&1
  echo "## cold scaffold (placebo)"
  python sim_practice.py --task-dir $TBASE/$T --cold \
    --policy-endpoint http://ava:4000/v1 --policy-model protolabs/smart --policy-key "$GATEWAY_API_KEY" \
    > logs/cold_$T.log 2>&1
  echo "## 3-arm transfer, 5 trials @ $(date +%H:%M:%S)"
  python run_transfer.py --task-dir $TBASE/$T \
    --sim-scaffold scaffolds/${T}.md --cold-scaffold scaffolds/${T}_cold.md \
    --trials 5 --port-offset ${PORT[$T]} --out results_${T}.json > logs/transfer_$T.log 2>&1
  echo "## $T result:"; tail -7 logs/transfer_$T.log
done
echo "###### ALL DONE @ $(date +%H:%M:%S) ######"
