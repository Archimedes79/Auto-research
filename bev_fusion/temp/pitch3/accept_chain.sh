#!/bin/bash
# the acceptance gates, one after the other, inside the clean export (CPU, nice); each waits for 19 GB of free RAM
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
H=$(cd "$REPO" && git rev-parse --short HEAD)
EXP=$REPO/temp/accept_$H
L=$REPO/bev_fusion/temp/pitch3
PY=$HOME/miniconda3/envs/gcar/bin/python
cd "$EXP/bev_fusion" || exit 1
rm -f $L/acc_*.log $L/acc_done.txt
for step in "door|tests/test_evaluate_door.py --real" "bev_free|tests/test_bev_free.py runs/t20_base_s0" "runner|-m runner Runner.predict runs/t20_base_s0 frames:scene-0103/3:val --device cpu --workers 0" "api|tests/test_evaluation_api.py --real runs/t20_base_s0"; do
  name=${step%%|*}; cmd=${step#*|}
  while true; do a=$(awk '/MemAvailable/ {print int($2/1000000)}' /proc/meminfo); [ "$a" -ge 19 ] && break; sleep 30; done
  date +%s > $L/acc_$name.start
  nice -n 10 $PY $cmd > $L/acc_$name.log 2>&1
  echo "exit $?" >> $L/acc_$name.log
done
echo "$H" > $L/acc_done.txt
