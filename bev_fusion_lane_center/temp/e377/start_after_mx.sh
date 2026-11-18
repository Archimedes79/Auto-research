#!/bin/bash
# E377: start the supervisor on slot 2 right after E376.mx_s1's training (its GPU dumps queue first: run.py's GPU queue is first come, first served)
cd "$(dirname "$0")/../.."
until grep -q "\[sup mx_s1\] training done" temp/e376/chain376.log 2>/dev/null; do sleep 20; done
sleep 45
echo "$(date '+%m-%d %H:%M:%S') [start] mx_s1 training done, starting sup377" >> temp/e377/chain377.log
bash temp/e377/sup377.sh
