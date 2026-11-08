#!/bin/bash
# scratch: rebuild the t16 report from the stored files (re-runnable; one CPU core)
cd /mnt/c/Auto-research || exit 1
/home/ml/miniconda3/envs/gcar/bin/python bev_fusion/temp/t16/build_report.py --out bev_fusion/results/t16/report.html
ls -la bev_fusion/results/t16/
