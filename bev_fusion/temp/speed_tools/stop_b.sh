#!/bin/bash
# stop MY speed.b (finished its 70 steps, stuck in interpreter teardown with 22 GB and swapping) and the waiting launcher of speed.c; nothing of another chat
cd /mnt/c/Auto-research/bev_fusion || exit 1
for id in speed.c speed.b; do
  for p in $(pgrep -f "tools/run.py --id $id "); do kill -9 $p && echo "launcher $id pid $p killed"; done
done
for p in $(pgrep -f "speed_run.sh speed temp/compose/speed_b"); do kill -9 $p && echo "wrapper pid $p killed"; done
for p in $(pgrep -f "train.py temp/compose/speed_b"); do kill -9 $p && echo "train pid $p killed"; done
sleep 5
free -m | sed -n 2,3p
pgrep -af "speed" | grep -v pgrep | cut -c1-120
