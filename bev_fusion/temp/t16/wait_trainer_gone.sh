#!/bin/bash
# scratch: wait until the hung trainer pid ($1) is gone, at most $2 seconds; then show the state
PID=$1
MAX=${2:-300}
t=0
while ps -p "$PID" >/dev/null 2>&1; do
  if [ "$t" -ge "$MAX" ]; then echo "still there after $t s"; break; fi
  sleep 5
  t=$((t + 5))
done
ps -o pid,stat,etime,cmd -p "$PID" | cut -c1-120
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
date +%T
