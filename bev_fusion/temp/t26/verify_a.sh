#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=~/miniconda3/envs/gcar/bin/python
echo "== check-cache-older (read only)"; nice -n 15 $PY training/lane_branch.py check-cache-older training/configs/t26_one.json 2>&1 | tail -4
echo "== check-cache (read only)"; nice -n 15 $PY training/lane_branch.py check-cache training/configs/t26_one.json 2>&1 | tail -4
echo "== startable selftest"; nice -n 15 $PY training/startable.py --selftest 2>&1 | tail -2
echo "== equivalence"; nice -n 15 $PY tests/test_trainkit_equivalence.py 2>&1 | grep -c "FAIL"
