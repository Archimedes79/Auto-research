#!/bin/bash
# prints every new line of the two chain logs (day14: E324 and the removal arms, day15: E292), once a minute; polled inside WSL (no tail -F)
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
declare -A n
for f in temp/day14/day14.log temp/day15/day15.log; do n[$f]=$(wc -l < $f); done
while true; do
  sleep 60
  for f in temp/day14/day14.log temp/day15/day15.log; do
    m=$(wc -l < $f)
    if [ "$m" -gt "${n[$f]}" ]; then tail -n +$((${n[$f]}+1)) $f | cut -c1-160 | sed "s#^#$(basename $f .log): #"; n[$f]=$m; fi
  done
done
