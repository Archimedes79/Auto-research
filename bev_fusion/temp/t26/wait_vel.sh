#!/bin/bash
# exits when t26.base_eval_vel has a done flag or has failed / been stopped
cd /mnt/c/Auto-research/bev_fusion
while true; do
  [ -e records/runs/t26.base_eval_vel/done ] && break
  s=$(python3 -c 'import json;print(json.load(open("records/runs/t26.base_eval_vel/run.json")).get("status"))' 2>/dev/null)
  case "$s" in failed|blocked|stalled|stopped|held) break;; esac
  sleep 120
done
echo "status=$s"; date; tail -n 3 records/runs/t26.base_eval_vel/run.log | cut -c1-200
