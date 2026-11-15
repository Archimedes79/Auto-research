#!/bin/bash
# prints the VmHWM (peak resident MB) and VmRSS of every running "python jtest_eval.py" process
for p in $(pgrep -f "python jtest_eval.py"); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-70)
  echo "pid $p | $c | HWM $(awk '/VmHWM/ {print int($2/1024)}' /proc/$p/status) MB | RSS $(awk '/VmRSS/ {print int($2/1024)}' /proc/$p/status) MB"
done
