#!/bin/bash
# gated recursion on nuScenes: train (quick) + sequence evaluation, seeds 0 and 1, GPU, one job at a time
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
BEST="--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --gate"
for s in 0 1; do
  $PY -u tests/test_flow_explains.py --quick --data nuscenes $BEST --seed $s > temp/checks/flow_explains_nusc_gate_s$s.log 2>&1
  $PY -u tests/test_flow_explains.py --data nuscenes --sequences --n-seq 24 --length 5 $BEST --seed $s > temp/checks/flow_sequences_nusc_gate_s$s.log 2>&1
done
