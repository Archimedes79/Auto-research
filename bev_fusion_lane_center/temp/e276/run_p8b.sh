#!/bin/bash
# scratch: P8b (coverage of the map pieces by tier / visibility) on the full dumps of B, CPU only, low priority
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do
  nice -n 12 bash tools/env/_py14.sh exp_coverage.py /mnt/c/Auto-research/bev_fusion_lane_center/temp/e276/predfull_e273_p_s$s.pkl /mnt/c/Auto-research/data/autogt/lane/0702a065c6 > temp/e276/p8b_s$s.log 2>&1
done
touch temp/e276/P8B_DONE
