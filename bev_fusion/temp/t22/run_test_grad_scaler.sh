#!/bin/bash
# scratch: runs tests/test_grad_scaler.py with the GPU visible (its cuda autocast checks need it; tiny tensors, well under a second of GPU) and prints its FAIL lines
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python tests/test_grad_scaler.py > temp/t22/test_grad_scaler.out 2>&1
echo "exit $?"
grep -n 'FAIL' temp/t22/test_grad_scaler.out | cut -c1-300
tail -2 temp/t22/test_grad_scaler.out | cut -c1-300
