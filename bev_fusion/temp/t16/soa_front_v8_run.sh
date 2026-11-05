#!/bin/bash
# run a python script of the bev_fusion folder under nice, 2 threads, CPU only, with the peak memory reported (usage: soa_front_v8_run.sh script args...)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
exec /usr/bin/time -f "[time] %e s wall, %U s user, peak RSS %M KB" nice -n 15 /home/ml/miniconda3/envs/gcar/bin/python "$@"
