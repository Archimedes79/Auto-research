#!/bin/bash
# scratch: paired scene bootstrap of the stand-in against the twin (the student effect, live code on both sides), CPU only, nice, 2 threads
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py compare runs/t22_twin_pair4 runs/t22_standin_pair4 --curves --fusion --n 1000 > temp/t22/compare_standin_vs_twin.log 2>&1
echo "exit $?" >> temp/t22/compare_standin_vs_twin.log
