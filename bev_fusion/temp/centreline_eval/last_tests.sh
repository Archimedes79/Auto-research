#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
PY="$HOME/miniconda3/envs/gcar/bin/python"
echo "=== test_frozen (expected to fail until the lead updates frozen.json)"; $PY tests/test_frozen.py 2>&1 | tail -8 | cut -c1-260
echo "=== bev_response selftest"; OMP_NUM_THREADS=2 $PY evaluation/bev_response.py --selftest 2>&1 | tail -3 | cut -c1-240
