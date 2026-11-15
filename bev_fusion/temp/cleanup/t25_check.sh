#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for a in ctl rec_r rec_scan rec_scan_mem; do
  echo "== $a"
  nice -n 15 $PY -m Training check ../bev_fusion/temp/cleanup/t25_${a}_job.json 2>&1 | grep -v '^WARNING' | tail -n 10 | cut -c1-260
done
