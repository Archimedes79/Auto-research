#!/bin/bash
# one job of this session at a time; each waits until the GPU has room for it (total stays < 11 GB, CLAUDE.md)
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
room() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt $((11000 - $1)) ]; }
wait_room() { until room $1 && sleep 30 && room $1; do sleep 60; done; }
BEST="--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --gate-doppler"
wait_room 1500
echo "gatedop s1 start $(date +%T)"
$PY -u tests/test_flow_explains.py --quick --data nuscenes $BEST --seed 1 > temp/checks/flow_explains_nusc_gatedop_s1.log 2>&1
$PY -u tests/test_flow_explains.py --data nuscenes --sequences --n-seq 24 --length 5 $BEST --seed 1 > temp/checks/flow_sequences_nusc_gatedop_s1.log 2>&1
for seed in 0 1; do
  for arm in "--label" "--boxes" "--boxes --maps"; do
    name=$(echo "$arm" | tr -d '-' | tr ' ' '_')
    case "$arm" in *maps*) need=3000 ;; *) need=1500 ;; esac
    wait_room $need
    echo "$name s$seed start $(date +%T)"
    $PY -u tests/test_flow_timed.py $arm --seed $seed > temp/checks/sup/full_${name}_s${seed}.log 2>&1
    echo "$name s$seed exit $? $(date +%T)"
  done
done
echo "queue finished $(date +%T)"
