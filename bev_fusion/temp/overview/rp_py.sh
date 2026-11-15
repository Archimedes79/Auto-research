#!/bin/bash
# usage: bash rp_py.sh script.py [args]   (CPU only, 2 threads, nice 15; script path relative to temp/overview)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd /mnt/c/Auto-research/bev_fusion
s=$1
shift
nice -n 15 $HOME/miniconda3/envs/gcar/bin/python temp/overview/$s "$@"
