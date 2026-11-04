#!/usr/bin/env bash
# the folder's other tests that touch the changed files; CPU only
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
PY="$HOME/miniconda3/envs/gcar/bin/python"
for t in "evaluation/metrics.py" "evaluation/bev_selftest.py" "evaluation/bev_gtcheck.py --selftest" "tests/test_orientation.py" "tests/test_layers.py" "tests/test_evaluate_door.py" "tests/test_bundle.py"; do
  echo "=== $t"
  OMP_NUM_THREADS=2 $PY $t 2>&1 | tail -4 | cut -c1-220
  echo "exit ${PIPESTATUS[0]}"
done
