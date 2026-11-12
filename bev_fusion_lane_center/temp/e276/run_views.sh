#!/bin/bash
# scratch: comparison pictures B | B + hard step | E277 soft-step head | arm t on six VAL frames, CPU, low priority
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 12 bash tools/env/_py14.sh compare_views.py /mnt/c/Auto-research/bev_fusion_lane_center/results/E277/figures 643,0,300,450,600,100 e273_p_s0 e273_p_s0+close e277_u_s0 e276_t_s0 > temp/e276/views.log 2>&1
touch temp/e276/VIEWS_DONE
