#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 12 bash tools/env/_py14.sh compare_views.py /mnt/c/Auto-research/bev_fusion_lane_center/results/E277/figures_test 643 e273_p_s0 e273_p_s0+close > temp/e276/views_test.log 2>&1
touch temp/e276/VIEWS_TEST_DONE
