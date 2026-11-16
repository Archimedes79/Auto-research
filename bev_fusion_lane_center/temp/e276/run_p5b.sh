#!/bin/bash
# scratch: P5b (link persistence) on the full dumps of B, CPU only, low priority; waits for the P10 job to finish first (shared box)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
until [ -f temp/e276/P10_DONE ]; do sleep 20; done
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do
  nice -n 12 bash tools/env/_py14.sh exp_temporal_links.py e273_p_s$s /mnt/c/Auto-research/data/autogt/lane/0702a065c6 > temp/e276/p5b_s$s.log 2>&1
done
touch temp/e276/P5B_DONE
