#!/bin/bash
# scratch: paired scene bootstrap, baseline t16_full_s0 (a) against the more-data arm t17_data (b), 1000 val frames, same GT signature
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
$PY evaluation/bev_suite.py compare runs/t16_full_s0 runs/t17_data --curves --fusion --velocity --n 1000 --boot 1000 > temp/t16/compare_t17.txt 2>&1
echo "exit $?"
wc -l temp/t16/compare_t17.txt
head -60 temp/t16/compare_t17.txt | cut -c1-200
