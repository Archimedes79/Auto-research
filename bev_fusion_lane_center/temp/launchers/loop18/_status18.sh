#!/bin/bash
# one-line status of every run of this loop (Git Bash or WSL); usage: _status18.sh [pattern]
X=/c/ML-Experiment/bev_fusion_lane_center/experiment_execution/runs
[ -d $X ] || X=/mnt/c/ML-Experiment/bev_fusion_lane_center/experiment_execution/runs
date '+%F %H:%M'
for d in $X/*${1}*/; do
  r=$(basename $d); s=$(grep -oE '"status": "[a-z_]+"' $d/run.json 2>/dev/null | head -1 | cut -d'"' -f4)
  echo "$r $s | $(tail -1 $d/run.log 2>/dev/null | cut -c1-70)"
done
