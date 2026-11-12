#!/bin/bash
# media of the junction report: GT-only figures, qualitative frames and scene videos (CPU). usage: make_media.sh <maps run e.g. e286_j1_s0> [pair dump run e.g. e287_p2_s0]
cd "$(dirname "$0")/../../tools/viz"
PRED=$1; PAIRS=$2
R=/mnt/c/Auto-research/bev_fusion_lane_center
M=$R/temp/jr/media
mkdir -p $M/gt $M/frames $M/videos
export CUDA_VISIBLE_DEVICES=""
PA=""; [ -n "$PAIRS" ] && PA="--pairs $PAIRS"
for s in scene-0096:10 scene-0221:12 scene-0017:30 scene-0100:15; do
  bash ../env/_py14.sh junction_view.py frame $s --out $M/gt --cam
done
for s in scene-0096:10 scene-0221:20 scene-0017:30 scene-0108:25 scene-0094:15 scene-0104:20; do
  bash ../env/_py14.sh junction_view.py frame $s --out $M/frames --pred $PRED $PA --cam
done
for sc in scene-0096 scene-0221 scene-0017 scene-0108; do
  bash ../env/_py14.sh junction_view.py scene $sc --out $M/videos --pred $PRED $PA --cam --every 1 --fps 2
done
echo MEDIA DONE
