#!/bin/bash
# prints one line whenever the status of a loop-19 run record changes (ok / failed / running ...); used as a Monitor source
cd "$(dirname "$0")/../.."
declare -A last
while true; do
  for d in experiment_execution/runs/E23[6-9].* experiment_execution/runs/E24[0-7].* experiment_execution/runs/E25*; do
    [ -f "$d/run.json" ] || continue
    id=$(basename "$d")
    s=$(grep -m1 '"status"' "$d/run.json" | sed 's/.*: *"\([a-z_]*\)".*/\1/')
    if [ "${last[$id]}" != "$s" ]; then
      [ -n "${last[$id]}" ] && echo "$(date +%H:%M) $id: ${last[$id]} -> $s"
      last[$id]=$s
    fi
  done
  sleep 45
done
