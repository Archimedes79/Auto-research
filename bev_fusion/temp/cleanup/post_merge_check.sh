#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
nice -n 15 $PY -m Training check ../bev_fusion/temp/cleanup/nol4_job.json 2>&1 | grep -v '^WARNING' | tail -n 12 | cut -c1-240
