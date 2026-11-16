#!/bin/bash
# scratch: stop the scheduler only (its jobs keep running); show what is pending
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
pkill -f "[l]aunch_chain.sh"
sleep 2
pgrep -af launch_chain | head -2
bash -n temp/e276/start_arm3.sh && echo arm3 ok
tail -4 temp/e276/chain.log
for r in E276.t_s1 E276.tq_s1 E278.qp_s0 E278.qp_s1; do echo -n "$r: "; grep -h '"status"' experiment_execution/runs/$r/run.json 2>/dev/null | head -1 | tr -d ' \n'; echo; done
date +%H:%M
