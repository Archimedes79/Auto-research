#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
echo "== gates (as run_gates.sh)"
nice -n 15 $PY evaluation/bev_junction_check.py | tail -1 | cut -c1-110
nice -n 15 $PY evaluation/soa_maps.py --selftest | tail -1 | cut -c1-60
nice -n 15 $PY evaluation/soa_maps.py --mutants | tail -1
echo "== reproduce 246"
nice -n 15 $PY evaluation/bev_junction_check.py --reproduce | tail -3
echo "== door, errors"
nice -n 15 $PY tests/test_evaluate_door.py | tail -1
nice -n 15 $PY evaluation/bev_errors.py --selftest | tail -1 | cut -c1-80
nice -n 15 $PY tests/test_evaluation_api.py | tail -1
echo "== bev_selftest"
nice -n 15 $PY evaluation/bev_selftest.py 2>&1 | grep -E "^(FAIL|bev selftest)|Traceback|Error" | cut -c1-140
echo FINALDONE
