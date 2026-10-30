#!/bin/bash
# t4_soa_front: RGB vs BGR input on 100 frames (decides the input order of the reference)
cd "$(dirname "$0")/../.."
for v in rgb bgr; do
  flag=""; [ "$v" = bgr ] && flag="--bgr"
  bash ../bev_fusion_lane_center/tools/env/_py14.sh temp/t4/dualbev_front_eval.py --n 100 --out runs/t4_chk_$v $flag 2>&1 | grep -v "^_py14" | tail -3
done
