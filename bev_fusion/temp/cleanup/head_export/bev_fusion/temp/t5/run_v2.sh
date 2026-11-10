#!/usr/bin/env bash
# t5 lane probe v2: fresh frames, features, five heads (50 epochs), evaluation - one after another (CPU, 2 threads).
# Usage (from the bev_fusion folder): bash temp/t5/run_v2.sh
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
run() { OMP_NUM_THREADS=2 nice -n 10 "$PY" -u temp/t5/lane_probe_v2.py "$@"; }
[ -f temp/t5/frames_v2.json ] || run frames || exit 1
run features > temp/t5/features_v2.log 2>&1 || { tail -5 temp/t5/features_v2.log; exit 1; }
tail -1 temp/t5/features_v2.log
for arm in ctx16 fpn16 res8 res4 rgb; do
  run train "$arm" > "temp/t5/train_v2_$arm.log" 2>&1 || { echo "FAIL $arm"; tail -5 "temp/t5/train_v2_$arm.log"; }
  tail -1 "temp/t5/train_v2_$arm.log"
done
run eval > temp/t5/eval_v2.log 2>&1
tail -25 temp/t5/eval_v2.log
