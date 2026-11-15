#!/bin/bash
# t26.base_eval: the stored base (runs/t20_base_s0 ckpt 6000, EMA, copied to temp/t26/base_run with its config: the live training code, the run's own code/ imports the removed autogt) on GT v13, VAL 1000, CPU
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
nice -n 15 ~/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py temp/t26/base_run --device cpu --workers 1 eval --n 1000
