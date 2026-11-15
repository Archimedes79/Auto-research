#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$(cd .. && pwd)"
PY=$HOME/miniconda3/envs/gcar/bin/python
run() { local name="$1"; shift; local t0=$(date +%s); if out=$("$@" 2>&1); then echo "PASS  $name ($(( $(date +%s) - t0 )) s) | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-110)"; else echo "FAIL  $name ($(( $(date +%s) - t0 )) s) | $(echo "$out" | grep -v -i warn | tail -3 | tr '\n' ' ' | cut -c1-260)"; fi; }
run "camera-off selftest" $PY training/feature_cache.py camera-off-selftest
run "grad scaler"         $PY tests/test_grad_scaler.py
run "clip split"          $PY tests/test_clip_split.py
run "host overhead"       $PY tests/test_host_overhead.py
run "evaluate door"       $PY tests/test_evaluate_door.py
run "splits"              $PY tests/test_splits.py
run "layers"              $PY tests/test_layers.py
run "velocity hurdle"     $PY tests/test_velocity_hurdle.py
run "orientation"         $PY tests/test_orientation.py
echo "done"
