#!/bin/bash
# E354: queue the four trainings (run.py FIFO, two GPU slots) and the two post jobs (run.py --after), detached from the calling shell.
# usage: launch_all.sh <K> <bs>     K / bs from the solo launch gate of the card (3 4)
cd "$(dirname "$0")/../.."
K=$1; BS=$2
for a in "r 0" "z 0" "r 1" "z 1"; do
  set -- $a
  setsid nohup bash temp/e354/start_arm.sh $1 $2 $K $BS > temp/e354/start_$1_s$2.out 2>&1 < /dev/null &
  sleep 5                                                   # the FIFO order of the tickets: r_s0, z_s0, r_s1, z_s1
done
for s in 0 1; do
  setsid nohup bash temp/e354/post_start.sh $s > temp/e354/post_start_s$s.out 2>&1 < /dev/null &
done
sleep 25
ps -eo pid,etimes,cmd | grep -E "run.py --id E354|jrecur_train" | grep -v grep | cut -c1-110
