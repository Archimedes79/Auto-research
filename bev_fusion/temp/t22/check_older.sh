#!/bin/bash
# scratch: compares the older-frame cache with the dataset's own images on 20 val samples (CPU, one process, low priority)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
nice -n 12 $HOME/miniconda3/envs/gcar/bin/python training/lane_branch.py check-cache-older runs/t22_B_s0/config.json --n 20 2>&1 | tail -3 | cut -c1-300
echo "exit ${PIPESTATUS[0]}"
du -sh data/cache/img_front/sweeps
find data/cache/img_front/sweeps -name '*.jpg' | wc -l
find data/cache/img_front/sweeps -name '*.tmp*' | wc -l
