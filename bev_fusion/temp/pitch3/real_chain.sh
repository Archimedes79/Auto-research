#!/bin/bash
# the real gates of the cleanup, one after the other (CPU, nice): after the quick battery has finished and 19 GB of RAM are free
cd "$(git rev-parse --show-toplevel)/bev_fusion"
L=temp/pitch3
while ! grep -q "^DONE" $L/gates_out.txt 2>/dev/null; do sleep 10; done
PY=$HOME/miniconda3/envs/gcar/bin/python
for step in "golden|evaluation/bev_selftest.py --golden" "bev_free|tests/test_bev_free.py" "door|tests/test_evaluate_door.py --real" "api|tests/test_evaluation_api.py --real runs/t20_base_s0"; do
  name=${step%%|*}; cmd=${step#*|}
  while true; do a=$(awk '/MemAvailable/ {print int($2/1000000)}' /proc/meminfo); [ "$a" -ge 19 ] && break; sleep 30; done
  date +%s > $L/real_$name.start
  nice -n 10 $PY $cmd > $L/real_$name.log 2>&1
  echo "exit $?" >> $L/real_$name.log
done
echo ALLDONE > $L/real_done.txt
