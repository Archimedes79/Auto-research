#!/bin/bash
# scratch: wait until the run record of run id $1 (records/runs/<id>/run.json) leaves the states running / waiting, or $2 minutes pass
# usage: wait_run_end.sh t17.data 400
cd /mnt/c/Auto-research/bev_fusion || exit 1
ID=$1
MAXMIN=${2:-600}
REC=records/runs/$ID/run.json
t=0
while [ "$t" -lt $((MAXMIN * 60)) ]; do
  st=$(grep -o '"status": "[a-z_]*"' "$REC" 2>/dev/null | head -1 | cut -d'"' -f4)
  if [ -n "$st" ] && [ "$st" != "running" ] && [ "$st" != "waiting" ]; then break; fi
  sleep 30
  t=$((t + 30))
done
echo "$ID status=$st after $((t / 60)) min at $(date +%T)"
grep -E '"exit"|"finished"' "$REC"
tail -3 "records/runs/$ID/run.log" | cut -c1-200
