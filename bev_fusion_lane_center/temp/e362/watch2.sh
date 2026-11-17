#!/bin/bash
# E362 watcher 2: progress every 10 minutes, new ROLLBACK / FAILED lines, and a STALL alert when a training log has not moved for 6 minutes; ends with the chain.
cd "$(dirname "$0")/../.."
n=0; seen=$(wc -l < temp/e362/chain.log)
while true; do
  c=$(wc -l < temp/e362/chain.log)
  if [ "$c" -gt "$seen" ]; then tail -n +$((seen+1)) temp/e362/chain.log | cut -c1-160; seen=$c; fi
  if tail -n 1 temp/e362/chain.log | grep -q "chain done\|not through"; then echo "WATCH END"; exit 0; fi
  n=$((n+1))
  for S in 0 1; do
    f=runs/e362_p4_s$S/log.txt
    [ -f $f ] || continue
    age=$(( $(date +%s) - $(stat -c %Y $f) ))
    [ -s temp/e362/train_s$S.rc ] || { [ $age -gt 360 ] && echo "STALL s$S: log not moved for $age s"; }
    [ $((n % 5)) -eq 1 ] && echo "s$S: $(grep ' step ' $f | tail -n 1 | cut -c1-60)"
  done
  for S in 0 1; do grep -E "ROLLBACK|Traceback|non-finite|FAILED" runs/e362_p4_s$S/log.txt | cut -c1-110 | sed "s/^/s$S ALERT: /"; done | sort -u > temp/e362/alerts.now2
  comm -13 <(sort temp/e362/alerts.seen2 2>/dev/null) temp/e362/alerts.now2
  sort -u temp/e362/alerts.now2 temp/e362/alerts.seen2 2>/dev/null > temp/e362/alerts.tmp2; mv temp/e362/alerts.tmp2 temp/e362/alerts.seen2
  sleep 120
done
