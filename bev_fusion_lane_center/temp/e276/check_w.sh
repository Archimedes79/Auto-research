#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
date +%H:%M
tail -3 temp/e276/chain.log
for k in 0 1; do
  last=$(ls -d runs/e279_w_s${k}_p* | sort | tail -1); echo "w_s$k $last: $(tail -1 $last/log.txt | cut -c1-150)"
  grep -h 'GATE' runs/e279_w_s${k}_p3/log.txt | cut -c1-260
done
for r in E279.w_s0 E279.w_s1; do echo -n "$r: "; grep -h '"status"' experiment_execution/runs/$r/run.json | head -1 | tr -d ' \n'; echo; done
