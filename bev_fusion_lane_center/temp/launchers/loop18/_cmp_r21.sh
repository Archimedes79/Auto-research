#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E213 / R2.1 comparison tables (failure series, hold variant): exact map, unregistered 1-3 m error, registered + trust gate, no map, wrong map (gated and not)
R=$BFLC
M=tables/e213.md
rm -f $R/runs/e17/$M
for p in dark blur cam_front3_off; do
  bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M \
    --entries exact=builder:e139_b/${p}_maxdecay_0.95_m0 U1=builder:e18_r21/${p}_maxdecay_0.95_U_s1 U2=builder:e139_b/${p}_maxdecay_0.95_m2 U3=builder:e18_r21/${p}_maxdecay_0.95_U_s3 \
      R1=builder:e18_r21/${p}_maxdecay_0.95_R62_s1 R2=builder:e18_r21/${p}_maxdecay_0.95_R62_s2 R3=builder:e18_r21/${p}_maxdecay_0.95_R62_s3 R2_thr75=builder:e18_r21/${p}_maxdecay_0.95_R75_s2 R2_notrust=builder:e18_r21/${p}_maxdecay_0.95_Rnotrust_s2 \
      none=builder:e139_b/${p}_maxdecay_0.95_mnone wrong=builder:e139_b/${p}_maxdecay_0.95_mwrong wrong_R62=builder:e18_r21/${p}_maxdecay_0.95_R62_wrong wrong_R75=builder:e18_r21/${p}_maxdecay_0.95_R75_wrong \
    --boot "R2,U2;R2,none;R2,exact;wrong_R62,wrong;wrong_R62,none;R1,U1;R3,U3"
done
