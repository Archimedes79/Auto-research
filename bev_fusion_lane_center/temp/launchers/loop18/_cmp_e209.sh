#!/bin/bash
# E209 tables: builders trained WITHOUT night / blur rows (B2 with the map, C2 without), evaluated on night and blur; references: champion without recursion, builders trained WITH all seven failure types (E139 B / C)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
M=tables/e209.md
rm -f $R/runs/e17/$M
for p in dark blur; do
  for v in single maxdecay_0.95; do
    echo "### $p $v"
    bash $R/tools/env/_py14.sh p17_compare.py --pert $p --md $M \
      --entries champ=builder:e132_norec/${p}_$v C_seen=builder:e139_c/${p}_${v}_mnone B_seen=builder:e139_b/${p}_${v}_m0 C2_unseen=builder:e142/${p}_${v}_C2none B2_nomap=builder:e142/${p}_${v}_B2none B2_exact=builder:e142/${p}_${v}_B2m0 \
      --boot "B2_exact,C2_unseen;B2_exact,champ;C2_unseen,champ;B2_exact,B2_nomap;B2_exact,B_seen;C2_unseen,C_seen"
  done
done
