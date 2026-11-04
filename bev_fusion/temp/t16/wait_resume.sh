#!/bin/bash
# scratch: wait until the second attempt of t17_data prints its first step line after the restart (or 6 minutes), then show the tail
cd /mnt/c/Auto-research/bev_fusion || exit 1
LOG=runs/t17_data/train.log
n0=$(grep -c "^step " "$LOG")
t=0
while [ "$t" -lt 360 ]; do
  n=$(grep -c "^step " "$LOG")
  if [ "$n" -gt "$n0" ]; then break; fi
  sleep 10
  t=$((t + 10))
done
tail -12 "$LOG" | cut -c1-220
date +%T
