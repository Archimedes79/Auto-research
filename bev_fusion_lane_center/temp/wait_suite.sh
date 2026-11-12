#!/bin/bash
# one-off: wait (at most 10 min) until the lane-suite run of the system given as $1 (default recall_sp0_dp) has ended
S=${1:-recall_sp0_dp}
for i in $(seq 1 100); do
  pgrep -f "lane_suite.py graphs:$S" >/dev/null || break
  sleep 6
done
pgrep -f "lane_suite.py graphs:$S" >/dev/null && echo still-running || echo done
