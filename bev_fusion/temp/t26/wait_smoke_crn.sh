#!/bin/bash
# exits when t26.ref_crn_smoke is no longer running
cd /mnt/c/Auto-research/bev_fusion
while true; do
  s=$(python3 -c 'import json;print(json.load(open("records/runs/t26.ref_crn_smoke/run.json")).get("status"))' 2>/dev/null)
  [ "$s" != "running" ] && [ "$s" != "waiting" ] && break
  sleep 20
done
echo "t26.ref_crn_smoke status=$s"; date
tail -n 40 records/runs/t26.ref_crn_smoke/run.log | cut -c1-240
