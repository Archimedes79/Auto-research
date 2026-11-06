#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
V=runs/t16_full_s0/eval/suite/qualitative/videos
for s in scene-0035 scene-0097 scene-0561; do
  /home/ml/miniconda3/envs/gcar/bin/python temp/t16/grab_frame.py $V/${s}_dashboard.mp4 temp/t16/dash_s6_$s.png
done
ls -la $V/*.mp4 | awk '{print $5, $9}'
