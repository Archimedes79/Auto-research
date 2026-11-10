#!/bin/bash
# wait until 19 GB of RAM are available (the box is shared), then run the real gate once; the result goes to temp/pitch2/api_real.log
cd "$(dirname "$0")/../.."
while true; do
  a=$(awk '/MemAvailable/ {print int($2/1000000)}' /proc/meminfo)
  if [ "$a" -ge 19 ]; then break; fi
  sleep 30
done
date +%s > temp/pitch2/api_real.start
nice -n 10 "$HOME/miniconda3/envs/gcar/bin/python" tests/test_evaluation_api.py --real runs/t20_base_s0 > temp/pitch2/api_real.log 2>&1
echo "exit $?" >> temp/pitch2/api_real.log
