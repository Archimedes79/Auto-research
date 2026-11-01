#!/bin/bash
# E238 tables: registered (R62) against unregistered (U2) and no map, three draws of the 2 m map error (d0 = stored E213.b draw, d1 / d2 = new), failure half.
cd "$(dirname "$0")/../.."
M=tables/e238.md
rm -f runs/e17/$M
for p in dark blur cam_front3_off; do
  bash tools/env/_py14.sh p17_compare.py --pert $p --md $M \
    --entries exact=builder:e140_gateB/${p}_gateB_m0 none=builder:e140_gateB/${p}_gateB_mnone \
      U2d0=builder:e140_gateB/${p}_gateB_m2 R2d0=builder:e18_r21b/${p}_gateB_R62_s2 \
      U2d1=builder:e19_e238/${p}_gateB_U2_d1 R2d1=builder:e19_e238/${p}_gateB_R62_d1 \
      U2d2=builder:e19_e238/${p}_gateB_U2_d2 R2d2=builder:e19_e238/${p}_gateB_R62_d2 \
    --boot "R2d1,U2d1;R2d2,U2d2;R2d1,none;R2d2,none;R2d0,U2d0"
done
cat runs/e17/$M
