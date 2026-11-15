#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
nice -n 15 $PY evaluation/bev_junction.py | tail -1
nice -n 15 $PY evaluation/soa_maps.py --selftest | tail -1 | cut -c1-90
nice -n 15 $PY evaluation/soa_maps.py --mutants | tail -1
nice -n 15 $PY evaluation/bev_selftest.py 2>&1 | grep -E "^(ok|FAIL|bev selftest)" | cut -c1-120
nice -n 15 $PY tests/test_evaluation_api.py | tail -1
echo FINALDONE
