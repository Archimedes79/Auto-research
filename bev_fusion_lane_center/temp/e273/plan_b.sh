#!/bin/bash
# attempt 7 = the set-attention link head (attempt 5) kept: restore its code and its phase 4 model, continue with phases 5-7 without stopping at the failed gate 4 (by hand: the complete set of numbers is wanted)
cd "$(dirname "$0")/../.."
git -C .. checkout 59ad3009 -- bev_fusion_lane_center/training/grid/qlinks.py bev_fusion_lane_center/training/grid/qgraph.py bev_fusion_lane_center/training/grid/qpredict.py
mv runs/e273_p_s0_p4 runs/_superseded/e273_p_s0_v6_p4
cp -r runs/_superseded/e273_p_s0_v5_p4 runs/e273_p_s0_p4
ls runs/e273_p_s0_p4 | head -5
git -C .. status --short bev_fusion_lane_center/training/grid | head
