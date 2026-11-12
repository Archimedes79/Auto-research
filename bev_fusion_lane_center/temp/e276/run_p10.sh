#!/bin/bash
# scratch: P10 (link thresholds on the graph numbers of B) on the full dumps, CPU only, low priority
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do
  nice -n 12 bash tools/env/_py14.sh exp_link_threshold.py /mnt/c/Auto-research/bev_fusion_lane_center/temp/e276/predfull_e273_p_s$s.pkl e273_p_s$s > temp/e276/p10_s$s.log 2>&1
done
touch temp/e276/P10_DONE
