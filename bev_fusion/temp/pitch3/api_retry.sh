#!/bin/bash
REPO=$(cd "$(dirname "$0")/../../.." && pwd); H=$(cd "$REPO" && git rev-parse --short HEAD)
EXP=$REPO/temp/accept_c4ec793f; L=$REPO/bev_fusion/temp/pitch3
cd "$EXP/bev_fusion" || exit 1
while true; do a=$(awk '/MemAvailable/ {print int($2/1000000)}' /proc/meminfo); [ "$a" -ge 20 ] && break; sleep 30; done
date +%s > $L/acc_api2.start
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python tests/test_evaluation_api.py --real runs/t20_base_s0 > $L/acc_api2.log 2>&1
echo "exit $?" >> $L/acc_api2.log
