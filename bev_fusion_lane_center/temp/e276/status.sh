#!/bin/bash
# scratch: one-screen state of the night (E275 launcher, E276 scheduler, run states, GPU)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
date +%H:%M:%S
echo "== e275 launch.log"; cat temp/e275/launch.log
echo "== e276 chain.log"; cat temp/e276/chain.log
echo "== GPU"; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "== run states"
for d in experiment_execution/runs/E275.* experiment_execution/runs/E276.*; do
  [ -d "$d" ] || continue
  echo -n "$(basename $d): "; grep -h '"status"' $d/run.json 2>/dev/null | head -1 | tr -d ' \n'; echo -n "  done="; [ -f $d/done ] && echo -n yes || echo -n no; echo
done
echo "== run.py processes"; pgrep -af "tools/run.py" | cut -c1-120
echo "== flags"; ls temp/e276 | grep -E "^(READY|STARTED|STAGE|VERDICT|CHAIN)" | tr '\n' ' '; echo
