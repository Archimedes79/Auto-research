#!/bin/bash
# who started the launchers of t10.dual_long_s0: their parent chain (pid, ppid, elapsed, command)
for pid in $(pgrep -f "tools/run.py --id t10.dual_long_s0"); do
  echo "launcher $pid:"
  p=$pid
  for i in 1 2 3 4 5 6 7 8; do
    line=$(ps -o pid=,ppid=,etime=,args= -p "$p" | cut -c1-200)
    [ -z "$line" ] && { echo "  (pid $p is gone)"; break; }
    echo "  $line"
    pp=$(ps -o ppid= -p "$p" | tr -d ' ')
    [ -z "$pp" ] || [ "$pp" = "0" ] || [ "$pp" = "1" ] && break
    p=$pp
  done
done
