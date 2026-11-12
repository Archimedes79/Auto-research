#!/bin/bash
# GT-only figures of the junction report (CPU, one thread, low priority)
cd "$(dirname "$0")/../../tools/viz"
R=/mnt/c/Auto-research/bev_fusion_lane_center
M=$R/temp/jr/media
mkdir -p $M/gt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for s in scene-0096:10 scene-0221:12 scene-0017:30 scene-0100:15; do
  nice -n 10 bash ../env/_py14.sh junction_view.py frame $s --out $M/gt --cam
done
echo GT MEDIA DONE
