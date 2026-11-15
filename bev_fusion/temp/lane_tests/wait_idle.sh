#!/bin/bash
# scratch: exits when no job of any chat runs (run.py, training, evaluation, qgraph, the t22 / E275 launchers) and the GPU is idle for 5 checks in a row (1 minute apart), or after 3 hours (then it says it is still busy).
# It only signals; the shutdown is done by hand after a last look.
PAT='tools/run\.py --id|training/train\.py|evaluation/evaluate\.py|bev_suite|qgraph_|trainsuite|start_A2|start_evalB|launch_scoring|launch_p\.sh|launch_window'
quiet=0
for i in $(seq 1 180); do
  n=$(ps -eo args | grep -E "$PAT" | grep -v grep | grep -v wait_idle | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$n" -eq 0 ] && [ "$mem" -lt 2000 ] && [ "$util" -lt 10 ]; then quiet=$((quiet+1)); else quiet=0; fi
  [ $quiet -ge 5 ] && { echo "IDLE: no job for 5 minutes ($(date +%T), gpu ${mem} MiB)"; exit 0; }
  sleep 60
done
echo "STILL BUSY after 3 hours ($(date +%T)): jobs=$n gpu=${mem} MiB"
exit 1
