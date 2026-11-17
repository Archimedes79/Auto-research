#!/bin/bash
# Close-out watcher 2: emits (a) E368 jobs gone / present changes, (b) report files changed, (c) new commits.
cd /c/Auto-research || exit 1
R=bev_fusion_lane_center/results/E2E_Champion_Report
last_j=""; last_f=""; last_c=""
for i in $(seq 1 340); do
  j=$(wsl bash -c "ps -eo cmd | grep -E 'run.py --id E368|jpast_train' | grep -v grep | wc -l" 2>/dev/null | tr -d '\r')
  if [ "$j" != "$last_j" ]; then echo "E368 processes: $j"; last_j="$j"; fi
  f=$(ls -l --time-style=+%H:%M:%S $R/*.html $R/*.zip 2>/dev/null | awk '{print $6, $7}' | tr '\n' ';')
  if [ "$f" != "$last_f" ]; then echo "REPORT FILES: $f"; last_f="$f"; fi
  c=$(git log --oneline -1 2>/dev/null | cut -c1-110)
  if [ "$c" != "$last_c" ]; then echo "HEAD: $c"; last_c="$c"; fi
  sleep 6
done
