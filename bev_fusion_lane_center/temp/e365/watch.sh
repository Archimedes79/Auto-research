#!/bin/bash
# E365 watcher: progress every 10 minutes, a STALL alert (log not moved for 7 minutes), relaunch / done lines of the supervisors; ends when both trainings are done.
cd "$(dirname "$0")/../.."
n=0; seen=$(wc -l < temp/e365/chain.log)
while true; do
  c=$(wc -l < temp/e365/chain.log)
  if [ "$c" -gt "$seen" ]; then tail -n +$((seen+1)) temp/e365/chain.log | cut -c1-140; seen=$c; fi
  if [ "$(grep -c 'training done' temp/e365/chain.log)" -ge 2 ] || grep -q "giving up" temp/e365/chain.log; then echo "WATCH END"; exit 0; fi
  n=$((n+1))
  for S in 0 1; do
    f=runs/e365_s1_s$S/log.txt
    [ -f $f ] || continue
    age=$(( $(date +%s) - $(stat -c %Y $f) ))
    if ! grep -q "done e365_s1_s$S" $f && [ $age -gt 420 ]; then echo "STALL s$S: log not moved for $age s"; fi
    if [ $((n % 5)) -eq 1 ]; then echo "s$S: $(grep ' step ' $f | tail -n 1 | cut -c1-100)"; fi
  done
  sleep 120
done
