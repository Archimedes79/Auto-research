#!/bin/bash
# Day chain 25 (2026-11-15 01:4x, lane lead): restart the TRAINING of E318.u_s1 (the stage that failed at 01:10 with ENOMEM in a data worker, step 69; id resumed by run.py, previous log kept).
# Why again: my temp/day22/day22.sh was overwritten by the lane experiments chat at 01:23 (name collision); when I killed that parent shell (pid 434595) at the chat's request, its children (start_pairs.sh 1
# and the launcher run.py --id E318.u_s1, still WAITING in the queue, nothing trained yet) died with it. This file only restarts them, detached in its own session (setsid -f), and then hands the
# post-processing to temp/day23/day23.sh (the lane experiments chat's file: it waits for the launcher, then runs post_pair, sigma_check, kperr, the maps flag and the bootstraps; if day23 already
# runs, flock makes this call exit 0). Recipe unchanged (temp/e318/start_pairs.sh 1, workers 3). D100: no host gate; one wait: MemAvailable >= 6 GB (the cause of the crash was memory).
# Never edit this script while it runs. Log temp/day25/day25.log, temp/e318/chain.log.
cd "$(dirname "$0")/../.."
D=temp/day25; mkdir -p $D
exec 9>/tmp/day25.lock
flock -n 9 || { echo "day25 already runs $(date)" >> $D/day25.log; exit 0; }
echo "day25 starts $(date)" >> $D/day25.log
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
L=temp/e318/chain.log
[ -f runs/e317_g_s1/model.pt ] || { echo "e318 seed 1 restart: the control runs/e317_g_s1 has no model $(date)" >> $L; exit 1; }
wait_start
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 6 ]; do echo "e318 seed 1 restart waits for MemAvailable >= 6 GB $(date)" >> $L; sleep 120; done
echo "e318 seed 1 RESTART (day25) of the training that failed at 01:10 (ENOMEM) and of the launcher killed at 01:3x $(date)" >> $L
bash temp/e318/start_pairs.sh 1 >> temp/e318/lane1.log 2>&1; XC=$?
echo "e318 seed 1 restart exit $XC $(date)" >> $L
echo "day25 training exit $XC $(date)" >> $D/day25.log
bash temp/day23/day23.sh >> temp/day23/day23_stdout.log 2>&1
echo "day25 done $(date)" >> $D/day25.log
