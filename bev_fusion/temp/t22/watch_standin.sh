#!/bin/bash
# scratch: one line per change of state of the two gate A (ii) evaluations (polls inside WSL; no tail -F)
cd /mnt/c/Auto-research/bev_fusion || exit 1
prev=""
while true; do
  cur=""
  for k in standin twin; do
    j=records/runs/t22.${k}_pair4/run.json
    s="none"; [ -f $j ] && s=$(python3 -c "import json;d=json.load(open('$j'));print(d.get('status'),d.get('exit_code'),d.get('duration_s'))" 2>/dev/null)
    l=runs/t22_${k}_pair4.log
    n=0; [ -f $l ] && n=$(wc -l < $l)
    e=0; [ -f runs/t22_${k}_pair4/eval/card.json ] && e=1
    cur="$cur | $k: $s log_lines=$n card=$e"
  done
  if [ "$cur" != "$prev" ]; then echo "$(date +%H:%M:%S) $cur"; prev="$cur"; fi
  case "$cur" in *"standin: none"*|*"twin: none"*|*"running"*|*"starting"*|*"waiting"*|*"None"*) ;; *) echo "both finished"; exit 0;; esac
  sleep 60
done
