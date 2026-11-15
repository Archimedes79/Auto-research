#!/bin/bash
cd /mnt/c/Auto-research/autoresearch
export AR_PROJECT=../bev_fusion PYTHONPATH=/mnt/c/Auto-research:/mnt/c/Auto-research/autoresearch OMP_NUM_THREADS=2
nice -n 15 ~/miniconda3/envs/gcar/bin/python -m Training check ../bev_fusion/temp/t26/job_smoke.json 2>&1 | tail -30 | cut -c1-600
