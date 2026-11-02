#!/bin/bash
# one line per finished run and per 20 steps; ends when no speed launcher is left
cd /mnt/c/Auto-research/bev_fusion || exit 1
last=""
while pgrep -f "[r]un.py --id speed" > /dev/null; do
  f=$(ls -t runs/speed_*/train.log 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    l=$(grep -E "^step [0-9]*[05]0:|exit |Error|Traceback|done" "$f" | tail -n 1 | cut -c1-150)
    cur="$f $l"
    if [ "$cur" != "$last" ]; then echo "$(date +%T) $cur"; last="$cur"; fi
  fi
  sleep 20
done
echo "$(date +%T) all speed launchers ended"
for x in a b c; do ls runs/speed_$x/.done 2>&1 | tail -n 1; done
