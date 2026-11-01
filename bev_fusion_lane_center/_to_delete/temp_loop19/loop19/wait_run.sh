#!/bin/bash
# wait (max $2 minutes, default 9) until run $1 has a final status; print the status and the tail of its run.log
R=experiment_execution/runs/$1
cd "$(dirname "$0")/../.."
for i in $(seq 1 $(( ${2:-9} * 12 ))); do
  s=$(grep -m1 '"status"' $R/run.json 2>/dev/null | sed 's/.*: *"\([a-z_]*\)".*/\1/')
  case "$s" in ok|failed|timeout|stalled|refused|blocked|gpu_busy) break ;; esac
  sleep 5
done
echo "status: $s"
tail -n ${3:-8} $R/run.log 2>/dev/null | cut -c1-260
