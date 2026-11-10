#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
P=~/miniconda3/envs/gcar/bin/python
$P tests/test_runner_predict.py runs/t20_base_s0 > temp/cleanup/test_pred.log 2>&1
echo "exit $?" >> temp/cleanup/test_pred.log
