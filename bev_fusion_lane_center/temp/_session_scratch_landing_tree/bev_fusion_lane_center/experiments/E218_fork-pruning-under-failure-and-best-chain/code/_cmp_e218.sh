#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E218: fork pruning (0.3) under failure; and the best chain (B + gate + registration + trust + pruning) against the same chain without pruning
R=$BFLC
M=tables/e218.md
rm -f $R/runs/e17/$M
for p in dark blur cam_front3_off; do
  for v in single maxdecay_0.95; do
    echo "### $p champion $v: no pruning (e132_norec) vs pruning 0.3"
    bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M --entries base=builder:e132_norec/${p}_$v fp03=builder:e18_fp/${p}_${v}_fp03 --boot "fp03,base"
  done
  echo "### $p best chain gateB: without / with pruning"
  bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M --entries exact=builder:e140_gateB/${p}_gateB_m0 exact_fp=builder:e18_best/${p}_gateB_fp03_m0 R75=builder:e18_r21b/${p}_gateB_R75_s2 R75_fp=builder:e18_best/${p}_gateB_fp03_R75_s2 none=builder:e140_gateB/${p}_gateB_mnone none_fp=builder:e18_best/${p}_gateB_fp03_mnone wrongR75=builder:e18_r21b/${p}_gateB_R75_wrong wrongR75_fp=builder:e18_best/${p}_gateB_fp03_R75_wrong --boot "R75_fp,R75;none_fp,none"
done
