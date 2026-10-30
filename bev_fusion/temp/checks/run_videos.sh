#!/bin/bash
# renders the GT driving videos of three mini scenes, one log per scene (run from anywhere; finds the folder itself)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p temp/checks/out_gt_video
for s in scene-0061 scene-1077 scene-0655; do
  ~/miniconda3/envs/gcar/bin/python temp/checks/video_gt_drive.py "$s" "temp/checks/out_gt_video/gt_drive_$s.mp4" > "temp/checks/out_gt_video/log_$s.txt" 2>&1
done
