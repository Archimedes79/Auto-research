#!/bin/bash
# one-off (E274): emit one line per new phase start / GATE / failure signature in the p logs; ends when both done flags exist
cd "$(dirname "$0")/../.."
prev=/tmp/monitor_p_prev.txt
[ -f $prev ] || : > $prev
for i in $(seq 1 3000); do
  cur=$(for s in 0 1; do for f in runs/e274_p_s${s}_p*/log.txt; do [ -f "$f" ] && grep -H -E "Traceback|Error|\bnan\b|NaN|Killed|OOM|GATE| phase [0-9]:|done [0-9]+$" "$f" | cut -c1-230; done; done | sort)
  echo "$cur" | comm -13 $prev - | grep -v '^$'
  echo "$cur" > $prev
  [ -f experiment_execution/runs/E274.p_s0/done ] && [ -f experiment_execution/runs/E274.p_s1/done ] && { echo "both p done flags"; exit 0; }
  sleep 60
done
