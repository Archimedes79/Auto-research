#!/usr/bin/env bash
# the evaluation selftest on the CPU, 2 threads (run from the bev_fusion folder)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
nice -n 10 /home/ml/miniconda3/envs/gcar/bin/python evaluation/bev_selftest.py 2>&1 | grep -v -i warn | tail -n 60
