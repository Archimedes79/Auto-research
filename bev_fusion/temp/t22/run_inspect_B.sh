#!/bin/bash
# scratch: runs inspect_B.py on the CPU with 2 threads
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python temp/t22/inspect_B.py 2>&1 | cut -c1-500
