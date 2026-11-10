#!/bin/bash
cd "$(dirname "$0")/../.."
PY="$HOME/miniconda3/envs/gcar/bin/python"
echo "== bev_selftest"; $PY evaluation/bev_selftest.py 2>&1 | tail -8
echo "== real golden"; OMP_NUM_THREADS=4 $PY tests/test_evaluate_door.py --real 2>&1 | tail -6
echo "== real sets"; OMP_NUM_THREADS=4 $PY tests/test_evaluate_door.py --sets 2>&1 | tail -6
echo "== all done"
