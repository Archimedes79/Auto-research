#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
for t in grid/jmaps.py grid/test_jmaps_dirx.py grid/jends.py; do bash tools/env/_py14.sh $t 2>&1 | tail -2; done
bash tools/env/_py14.sh jmaps_eval.py selftest 2>&1 | tail -3
