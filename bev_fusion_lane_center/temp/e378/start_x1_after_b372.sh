#!/bin/bash
# E378.x_s1 on GPU slot 1 after the S4 chain: waits until dumps372 has run E372.dump_b_s1 (or b_s1 was KILLED), then runs the
# unchanged x_s1 supervisor temp/e378/sup378b.sh (train 12000 steps via run.py --gpu, then its dumps and CPU read-out).
# Refuses to start after 05:00 (the night window's last start); run.py refuses a second E378.x_s1 while one runs.
cd "$(dirname "$0")/../.."
L=temp/e378/chain378.log
say() { echo "$(date '+%m-%d %H:%M:%S') [start_x1] $*" >> $L; }
say "waiting for E372.dump_b_s1 (slot 1)"
until grep -qE "\[dumps372\] (b_s1 live dump exit|b_s1 KILLED)" temp/e372/chain372.log; do sleep 30; done
H=$(date +%H%M)
if [ "$H" -ge 0500 ] && [ "$H" -lt 2200 ]; then say "slot free only at $H: after the 05:00 start limit, not started"; exit 0; fi
say "slot 1 free: starting sup378b (x_s1)"
bash temp/e378/sup378b.sh
say "sup378b returned $?"
