#!/usr/bin/env bash
# t5 lane probe: after the feature extraction, train the five probe heads one after another and evaluate (CPU, 2 threads).
# Usage (from the bev_fusion folder): bash temp/t5/run_arms.sh
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
while pgrep -f "[l]ane_probe.py features" > /dev/null; do sleep 60; done
[ "$(cat temp/t5/feats/done.txt 2>/dev/null)" = "900" ] || { echo "features incomplete"; exit 1; }
for arm in ctx16 fpn16 res8 res4 rgb; do
  OMP_NUM_THREADS=2 nice -n 10 "$PY" -u temp/t5/lane_probe.py train "$arm" > "temp/t5/train_$arm.log" 2>&1 || { echo "FAIL $arm"; tail -5 "temp/t5/train_$arm.log"; }
  tail -1 "temp/t5/train_$arm.log"
done
OMP_NUM_THREADS=2 nice -n 10 "$PY" -u temp/t5/lane_probe.py eval > temp/t5/eval.log 2>&1
tail -20 temp/t5/eval.log
