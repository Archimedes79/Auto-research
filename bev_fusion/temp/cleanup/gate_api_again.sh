#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
s=$(date +%s)
nice -n 15 ~/miniconda3/envs/gcar/bin/python tests/test_evaluation_api.py --real > /tmp/real_gates_mine/evaluation_api_real2.log 2>&1; rc=$?
echo "evaluation_api_real (second try) rc=$rc $(( $(date +%s) - s )) s | $(grep -v '^WARNING' /tmp/real_gates_mine/evaluation_api_real2.log | tail -n 1 | cut -c1-200)" >> /tmp/real_gates_mine/summary.txt
