#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
awk '/== verdict277 for E279/ {f=1} f' temp/e276/verdicts.log | head -80
python3 temp/e276/summarise_e279.py | grep -E "run|e273_p_s1 |e279_w|close1.5_t0.5 "
ls temp/e276 | grep -E "CHAIN"
