#!/bin/bash
# wait until every given run id has a done flag or a failed status; prints one line per id when it ends
cd "$(dirname "$0")/../.."
for id in "$@"; do
  while true; do
    d=experiment_execution/runs/$id
    if [ -f $d/done ]; then echo "$(date +%T) $id done"; break; fi
    if [ -f $d/run.json ] && grep -q '"status": "\(failed\|stalled\|timeout\|blocked\|gpu_busy\)"' $d/run.json; then echo "$(date +%T) $id ended: $(grep -o '"status": "[a-z_]*"' $d/run.json)"; break; fi
    sleep 20
  done
done
