#!/bin/bash
# E378: re-run x_s0's CPU read-out under the same ids once the killed launcher's heartbeat is stale (run.py: 300 s)
cd "$(dirname "$0")/../.."
sleep 360
echo "$(date '+%m-%d %H:%M:%S') [reread x_s0] starting the CPU read-out again" >> temp/e378/chain378.log
bash temp/e378/read378.sh cpu x_s0 >> temp/e378/read_x_s0.out 2>&1
