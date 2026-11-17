#!/bin/bash
# Close-out watcher: emits a line when (a) an E368 run.py job appears, (b) a commit mentioning "E375 result" appears.
cd /c/Auto-research || exit 1
seen_r=""
seen_c=""
for i in $(seq 1 340); do
  r=$(wsl bash -c "ps -eo cmd | grep -E 'run.py --id E368' | grep -v grep | cut -c1-90" 2>/dev/null | tr -d '\r')
  if [ -n "$r" ] && [ "$r" != "$seen_r" ]; then echo "E368 JOB: $r"; seen_r="$r"; fi
  c=$(git log --oneline -12 2>/dev/null | grep -i -E "E375 (result|verdict)|E375:.*(ADOPT|NO)" | head -1)
  if [ -n "$c" ] && [ "$c" != "$seen_c" ]; then echo "E375 COMMIT: $c"; seen_c="$c"; fi
  sleep 5
done
