#!/bin/bash
# E213.b tables: registration + trust gate in front of the best chain of loop 17 (map builder B + learned gate; failure half, series gateB)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
M=tables/e213b.md
rm -f $R/runs/e17/$M
for p in dark blur cam_front3_off; do
  bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M \
    --entries exact=builder:e140_gateB/${p}_gateB_m0 U1=builder:e140_gateB/${p}_gateB_m1 U2=builder:e140_gateB/${p}_gateB_m2 none=builder:e140_gateB/${p}_gateB_mnone wrong=builder:e140_gateB/${p}_gateB_mwrong \
      R1=builder:e18_r21b/${p}_gateB_R62_s1 R2=builder:e18_r21b/${p}_gateB_R62_s2 R3=builder:e18_r21b/${p}_gateB_R62_s3 R2_thr75=builder:e18_r21b/${p}_gateB_R75_s2 wrong_R62=builder:e18_r21b/${p}_gateB_R62_wrong wrong_R75=builder:e18_r21b/${p}_gateB_R75_wrong \
    --boot "R2,U2;R2,none;R2,exact;R2_thr75,none;wrong_R75,wrong;wrong_R75,none;R1,U1"
done
