#!/bin/bash
# scratch: rebuild the big t16 report from the stored files (one CPU core, low priority)
cd /mnt/c/Auto-research || exit 1
nice -n 15 /home/ml/miniconda3/envs/gcar/bin/python bev_fusion/temp/t16/build_big_report.py "$@"
ls -la bev_fusion/results/t16/ | head -20
