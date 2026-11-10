#!/bin/bash
# scratch: evaluation of the stage-B run t22.B_s0 (VAL 1000 frames, same flags as the gate A (ii) evaluations); waits until no other GPU job runs (D57: the lane work goes first) and stays inside the night window; diagnostic through the launcher (no card, never feeds a verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PY=$HOME/miniconda3/envs/gcar/bin/python
quiet=0
prev=""
while true; do
  h=$(date +%H%M)
  if [ "$h" -ge 0645 ] && [ "$h" -lt 2200 ]; then echo "$(date +%H:%M:%S) window closed (06:45 - 22:00), not started"; exit 3; fi
  others=$(ps -eo args | grep -E 'tools/run\.py --id|training/train\.py|evaluation/evaluate\.py' | grep -v grep | grep -v start_evalB | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$others" -eq 0 ] && [ "$mem" -lt 1800 ]; then quiet=$((quiet+1)); else quiet=0; fi
  cur="other_jobs=$others gpu_mb=$mem quiet=$quiet"
  if [ "$cur" != "$prev" ]; then echo "$(date +%H:%M:%S) $cur"; prev="$cur"; fi
  [ $quiet -ge 3 ] && break
  sleep 30
done
echo "$(date +%H:%M:%S) GPU quiet, starting the evaluation"
python3 ../autoresearch/tools/run.py --id t22.B_s0.eval --gpu --gpu-max-mb 3000 --split val --no-card -- \
  bash -c "$PY -u evaluation/evaluate.py runs/t22_B_s0 --quant --curves --fusion --velocity --n 1000 >> runs/t22_B_s0_eval.log 2>&1"
echo "$(date +%H:%M:%S) run.py exit $?"
