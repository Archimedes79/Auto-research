#!/bin/bash
cd /mnt/c/Auto-research || exit 1
T=bev_fusion/temp/critic_coverage
PY=~/miniconda3/envs/gcar/bin/python
PYTHONPATH=$PWD/$T/old OMP_NUM_THREADS=1 nice -n 15 $PY $T/gen_dump.py $T/dump_old.pkl 14 2>&1 | tail -5
PYTHONPATH=$PWD OMP_NUM_THREADS=1 nice -n 15 $PY $T/gen_dump.py $T/dump_new.pkl 14 2>&1 | tail -5
