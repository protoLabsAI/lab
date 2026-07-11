#!/bin/bash
set -e
cd /home/ava/dev/lab/experiments/agentworld/scaffold
set -a; source /home/ava/dev/lab/evals/.env; set +a
source /home/ava/dev/lab/.venv/bin/activate
TASK=../../../evals/claw/claw-eval/tasks/T104_packet_decoder
echo "### PHASE 1: sim-practice T104 @ $(date +%H:%M:%S)"
python sim_practice.py --task-dir $TASK \
  --policy-endpoint http://ava:4000/v1 --policy-model protolabs/smart --policy-key "$GATEWAY_API_KEY" \
  --aw-endpoint http://localhost:8010/v1 --episodes 1 --max-turns 12 > sim_practice_T104.log 2>&1
echo "### scaffold written; PHASE 2: transfer (baseline vs treatment, 2 trials) @ $(date +%H:%M:%S)"
python run_transfer.py --task-dir $TASK \
  --scaffold scaffolds/T104_packet_decoder.md --trials 2 --port-offset 500 > transfer_T104.log 2>&1
echo "### DONE @ $(date +%H:%M:%S)"
tail -8 transfer_T104.log
