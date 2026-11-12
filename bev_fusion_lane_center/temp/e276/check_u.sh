#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
date +%H:%M
tail -4 temp/e276/chain.log
for k in 0 1; do
  echo "== u_s$k"; grep -h 'GATE' runs/e277_u_s${k}_p3/log.txt | cut -c1-330
  last=$(ls -d runs/e277_u_s${k}_p* | sort | tail -1); echo "$last: $(tail -1 $last/log.txt | cut -c1-170)"
done
