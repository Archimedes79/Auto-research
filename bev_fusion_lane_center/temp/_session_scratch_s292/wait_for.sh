#!/bin/bash
# usage: wait_for.sh LOGFILE PATTERN [MAX_SECONDS]  -> waits until PATTERN appears in LOGFILE, then prints the tail
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
f="$1"; pat="$2"; mx="${3:-1800}"; n=0
while [ $n -lt $mx ]; do
  if grep -q -- "$pat" "$f" 2>/dev/null; then break; fi
  n=$((n+10)); sleep 10
done
date
tail -12 "$f" | cut -c1-260
