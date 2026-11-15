#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/temp/t25_review
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
{
  nice -n 15 $PY equiv_dump.py head/bev_fusion/training head.pt &&
  nice -n 15 $PY equiv_dump.py ../../training live.pt &&
  $PY equiv_cmp.py head.pt live.pt
} > equiv.log 2>&1
echo "exit $?" >> equiv.log
