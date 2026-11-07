#!/bin/bash
# scratch: exits (one notification) when no job of the lane project runs on this machine for 15 minutes in a row (qgraph training / evaluation, the E274 launchers, run.py of the lane project).
# Only a signal to me: the flag GO is created after the lane chat has said its queue is done, never by this script.
while true; do
  idle=0
  while [ $idle -lt 15 ]; do
    if pgrep -f "qgraph_|launch_p\.sh|launch_window|start_arm|AR_PROJECT=../bev_fusion_lane|--id E27" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
    sleep 60
  done
  echo "no lane job for 15 minutes: $(date)"
  exit 0
done
