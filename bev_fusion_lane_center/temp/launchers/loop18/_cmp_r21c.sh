#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E213.c: cost of the registration + trust gate on an exact map (failure half)
R=$BFLC
M=tables/e213c.md
rm -f $R/runs/e17/$M
for p in dark blur cam_front3_off; do
  echo "### $p builder B hold"
  bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M --entries exact=builder:e139_b/${p}_maxdecay_0.95_m0 none=builder:e139_b/${p}_maxdecay_0.95_mnone R62=builder:e18_r21c/${p}_maxdecay_0.95_R062_s0 R75=builder:e18_r21c/${p}_maxdecay_0.95_R075_s0 --boot "R75,exact;R62,exact"
  echo "### $p builder B + gate"
  bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M --entries exact=builder:e140_gateB/${p}_gateB_m0 none=builder:e140_gateB/${p}_gateB_mnone R62=builder:e18_r21c/${p}_gateB_R062_s0 R75=builder:e18_r21c/${p}_gateB_R075_s0 --boot "R75,exact;R62,exact"
done
