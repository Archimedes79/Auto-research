#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
n=0
while [ $n -lt 120 ]; do
  if grep -q "smoke exit" temp/_session_scratch_s292/logs/e351_smoke.log 2>/dev/null; then break; fi
  n=$((n+1))
  sleep 10
done
date
tail -30 temp/_session_scratch_s292/logs/e351_smoke.log | cut -c1-300
ls experiment_execution/runs/E351.smoke_g1/
