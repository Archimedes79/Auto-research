#!/bin/bash
# E367 watcher: new supervisor lines, a progress line per running training every 10 minutes, a STALL alert (log not moved for 7 minutes); ends when 7 trainings are done or a supervisor gives up.
cd "$(dirname "$0")/../.."
n=0; seen=$(wc -l < temp/e367/chain.log)
while true; do
  c=$(wc -l < temp/e367/chain.log)
  if [ "$c" -gt "$seen" ]; then tail -n +$((seen+1)) temp/e367/chain.log | grep -v "supervisor starts" | cut -c1-140; seen=$c; fi
  if [ "$(grep -c 'training done' temp/e367/chain.log)" -ge 7 ] || grep -q "giving up" temp/e367/chain.log; then echo "WATCH END"; exit 0; fi
  n=$((n+1))
  for f in runs/e367_*/log.txt; do
    [ -f "$f" ] || continue
    r=$(basename "$(dirname "$f")")
    grep -q "^[0-9:]* done $r\$" "$f" && continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [ $age -gt 420 ]; then echo "STALL $r: log not moved for $age s"; fi
    if [ $((n % 5)) -eq 1 ]; then echo "$r: $(grep ' step ' "$f" | tail -n 1 | cut -c1-90)"; fi
  done
  sleep 120
done
