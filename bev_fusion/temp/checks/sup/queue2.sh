#!/bin/bash
# after queue.sh: the correlation arm of the timed-velocity test, quick first (label / boxes / boxes + maps), then full
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
room() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt $((11000 - $1)) ]; }
wait_room() { until room $1 && sleep 30 && room $1; do sleep 60; done; }
while pgrep -f "[q]ueue\.sh" > /dev/null; do sleep 60; done
for arm in "--label --corr" "--boxes --corr" "--boxes --maps --corr"; do
  name=$(echo "$arm" | tr -d '-' | tr ' ' '_')
  case "$arm" in *maps*) need=3000 ;; *) need=1500 ;; esac
  wait_room $need
  echo "quick $name start $(date +%T)"
  $PY -u tests/test_flow_timed.py $arm --quick > temp/checks/sup/quick_${name}.log 2>&1
done
for seed in 0 1; do
  for arm in "--label --corr" "--boxes --corr"; do
    name=$(echo "$arm" | tr -d '-' | tr ' ' '_')
    wait_room 1500
    echo "full $name s$seed start $(date +%T)"
    $PY -u tests/test_flow_timed.py $arm --seed $seed > temp/checks/sup/full_${name}_s${seed}.log 2>&1
  done
done
echo "queue2 finished $(date +%T)"
