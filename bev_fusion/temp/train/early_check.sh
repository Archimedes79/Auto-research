#!/usr/bin/env bash
# One-off: early check of the combined run's latest checkpoint (200 val samples spread over all scenes, niced) and the
# vehicle heading diagnostic (full / axis error, flips) of its scene records.
cd "$(dirname "$0")/../.." || exit 1
nice -n 15 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/t1_r3_combined --workers 2 eval --n 200 > temp/train/early_check.log 2>&1
tail -3 temp/train/early_check.log
grep -E "det/dt\+0.00/(mean|vehicle)/all/(mAP|ap@2|ate|ase|aoe|ave)|detr/dt\+0.00/vehicle/all/(ap@2|ate|aoe)|seg_fov/seg/dt\+0.00/(drivable_area/all/all/iou@0|lane_divider/all/all/iou@1)" runs/t1_r3_combined/eval/table.txt
nice -n 15 ~/miniconda3/envs/gcar/bin/python temp/train/yaw_check.py t1_r3_combined 2>&1 | head -4
