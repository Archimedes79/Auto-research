#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
$HOME/miniconda3/envs/gcar/bin/python tests/test_student_chain.py > temp/t22/chain_test.log 2>&1
echo "exit $?" >> temp/t22/chain_test.log
