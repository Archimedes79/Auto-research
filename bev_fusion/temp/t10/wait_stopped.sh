#!/bin/bash
# wait (max 3 minutes) for the STOPPED marker of t10_dual_long_s0, then show what the stop left behind
cd "$(dirname "$0")/../../runs/t10_dual_long_s0" || exit 1
for i in $(seq 1 60); do
  [ -f STOPPED ] && break
  sleep 3
done
date +%H:%M:%S
ls -la --time-style=+%H:%M:%S STOPPED CONTROL ckpt_*.pt 2>&1 | cut -c1-130
echo "---- STOPPED:"
cat STOPPED 2>/dev/null | head -5
echo "---- train.log tail:"
tail -3 train.log | cut -c1-200
echo "---- processes of the run:"
pgrep -af "t10_dual_long_s0/code/train.py" | cut -c1-120 | head -3
