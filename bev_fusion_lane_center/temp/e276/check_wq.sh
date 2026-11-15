#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
date +%H:%M
tail -4 temp/e276/chain.log
for r in E279.w_s0 E279.w_s1 E279.wq_s0 E279.wq_s1; do echo -n "$r: "; grep -h '"status"' experiment_execution/runs/$r/run.json 2>/dev/null | head -1 | tr -d ' \n'; [ -f experiment_execution/runs/$r/done ] && echo -n " done"; echo; done
ls runs/e279 2>/dev/null
ls temp/e276 | grep -E "CHAIN|VERDICT"
