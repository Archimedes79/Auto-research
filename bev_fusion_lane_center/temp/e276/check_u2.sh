#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for r in E277.u_s0 E277.u_s1 E277.uq_s0 E278.qp_s1; do echo -n "$r: "; grep -h '"status"' experiment_execution/runs/$r/run.json 2>/dev/null | head -1 | tr -d ' \n'; [ -f experiment_execution/runs/$r/done ] && echo -n " done" ; echo; done
for k in 0 1; do echo "== u_s$k final gate"; grep -h 'GATE' runs/e277_u_s${k}_p7/log.txt | tail -1 | cut -c1-1800; tail -3 runs/e277_u_s${k}_p7/log.txt | cut -c1-220; done
tail -3 temp/e276/chain.log
