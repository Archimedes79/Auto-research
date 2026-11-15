#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
for i in $(seq 1 40); do
  [ -s records/runs/t26.ref_crn_smoke/run.log ] && break
  sleep 6
done
python3 -c 'import json;d=json.load(open("records/runs/t26.ref_crn_smoke/run.json"));print({k:d.get(k) for k in ("status","exit_code","commit","tag")})'
tail -n 14 records/runs/t26.ref_crn_smoke/run.log 2>/dev/null | cut -c1-220
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
