#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
for i in $(seq 1 50); do
  s=$(python3 -c 'import json;print(json.load(open("records/runs/t26.base_eval/run.json")).get("status"))')
  [ "$s" != "waiting" ] && break
  sleep 6
done
echo "status=$s"; tail -n 2 records/runs/t26.base_eval/heartbeat
tail -n 3 records/runs/t26.base_eval/run.log 2>/dev/null | cut -c1-200
ps -eo pid,etime,args | grep -E "bev_suite|run.py --id t26" | grep -v grep | cut -c1-170
