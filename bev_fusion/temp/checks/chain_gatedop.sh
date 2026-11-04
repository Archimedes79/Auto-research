#!/bin/bash
# gated recursion WITH the Doppler residual as gate input, nuScenes: train (quick) + sequence evaluation, seeds 0 and 1.
# Waits until no other training runs (another session's t4 runs used 10 GB of the GPU on 2026-10-30).
cd "$(dirname "$0")/../.."
free_now() { ! pgrep -f "[r]un_cfg.sh|[t]rain\.py" > /dev/null && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 4000 ]; }
until free_now && sleep 60 && free_now; do sleep 60; done
echo "start $(date +%T)"
PY=~/miniconda3/envs/gcar/bin/python
BEST="--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --gate-doppler"
for s in 0 1; do
  $PY -u tests/test_flow_explains.py --quick --data nuscenes $BEST --seed $s > temp/checks/flow_explains_nusc_gatedop_s$s.log 2>&1
  $PY -u tests/test_flow_explains.py --data nuscenes --sequences --n-seq 24 --length 5 $BEST --seed $s > temp/checks/flow_sequences_nusc_gatedop_s$s.log 2>&1
done
echo "done $(date +%T)"
