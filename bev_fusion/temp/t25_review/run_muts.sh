#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
: > temp/t25_review/muts.log
for m in "$@"; do
  nice -n 15 $PY temp/t25_review/mut_run.py $m > temp/t25_review/mut_$m.log 2>&1
  echo "== $m: $(grep -c 'FAIL' temp/t25_review/mut_$m.log) FAIL lines; $(grep 'FAIL\|EXIT' temp/t25_review/mut_$m.log | head -4 | cut -c1-170 | tr '\n' '|')" >> temp/t25_review/muts.log
done
echo finished >> temp/t25_review/muts.log
