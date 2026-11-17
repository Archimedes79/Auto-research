#!/bin/bash
# E373 trainings / read-outs and the E370 fixtures: new lines of both chain logs, a STALL alert (training log not moved for 7 minutes); ends after both E373 read-outs.
cd "$(dirname "$0")/../.."
touch temp/e373/chain.log
s3=$(wc -l < temp/e373/chain.log); s0=$(wc -l < temp/e370/chain.log)
while true; do
  c3=$(wc -l < temp/e373/chain.log); c0=$(wc -l < temp/e370/chain.log)
  if [ "$c3" -gt "$s3" ]; then tail -n +$((s3+1)) temp/e373/chain.log | grep -E "done|exit|giving" | cut -c1-110; s3=$c3; fi
  if [ "$c0" -gt "$s0" ]; then tail -n +$((s0+1)) temp/e370/chain.log | grep -E "fixture|exit" | cut -c1-110; s0=$c0; fi
  for f in runs/e373_cb_s*/log.txt; do
    [ -f "$f" ] || continue
    grep -q "done e373" "$f" && continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [ $age -gt 420 ]; then echo "STALL $f $age s"; fi
  done
  if [ "$(grep -c 'read exit' temp/e373/chain.log)" -ge 2 ]; then echo END; exit 0; fi
  sleep 60
done
