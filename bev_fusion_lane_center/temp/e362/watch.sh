#!/bin/bash
# E362 watcher: prints new chain.log lines, a progress line every 10 minutes, and ROLLBACK / Traceback / non-finite lines; ends when the chain is done.
cd "$(dirname "$0")/../.."
seen=0; n=0
while true; do
  c=$(wc -l < temp/e362/chain.log)
  if [ "$c" -gt "$seen" ]; then tail -n +$((seen+1)) temp/e362/chain.log | cut -c1-160; seen=$c; fi
  if grep -q "chain done\|is not through\|exists" temp/e362/chain.log; then echo "WATCH END"; exit 0; fi
  n=$((n+1))
  if [ $((n % 5)) -eq 1 ]; then
    for S in 0 1; do f=runs/e362_p4_s$S/log.txt; [ -f $f ] && echo "s$S: $(grep ' step ' $f | tail -n 1 | cut -c1-70) | $(tail -n 1 $f | cut -c1-90)"; done
  fi
  for S in 0 1; do f=runs/e362_p4_s$S/log.txt; [ -f $f ] && grep -E "ROLLBACK|Traceback|non-finite|unknown error|Killed" $f | cut -c1-140 | sed "s/^/s$S ALERT: /"; done | sort -u > temp/e362/alerts.now
  comm -13 <(sort temp/e362/alerts.seen 2>/dev/null) temp/e362/alerts.now; cat temp/e362/alerts.now temp/e362/alerts.seen 2>/dev/null | sort -u > temp/e362/alerts.tmp; mv temp/e362/alerts.tmp temp/e362/alerts.seen
  sleep 120
done
