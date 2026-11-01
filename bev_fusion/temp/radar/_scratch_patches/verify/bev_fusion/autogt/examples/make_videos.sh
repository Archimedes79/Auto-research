#!/bin/bash
# the three example videos (H.264, ~20 s each): run from anywhere
cd "$(dirname "$0")/../.." || exit 1
for s in scene-0061 scene-1077 scene-0655; do
  ~/miniconda3/envs/gcar/bin/python -m autogt video "$s" "autogt/examples/gt_drive_$s.mp4" > "autogt/examples/log_$s.txt" 2>&1
done
