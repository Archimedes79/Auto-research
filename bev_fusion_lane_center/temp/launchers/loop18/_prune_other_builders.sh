#!/bin/bash
# robustness of fork pruning on independently trained builders of loop 16 (replicate s2 of the recipe, plain continuation c1, time recursion t1, junction head j1 at 288 / 144)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
for n in e123_D7_r3_s2 e123_D7_r3_c1 e123_D7_r3_t1 e123_D7_r3_g0; do
  bash $R/tools/env/_py14.sh p18_variant_prune.py --name $n --thr 0,0.3,0.4 --json_thr 0.3 --tag other_${n: -2} 2>&1 | tail -3
done
