#!/bin/bash
# Night trigger (D107, 2026-11-15 22:40): everything of mine that waits for the END of the E354 seed-1 trainings, in the order the lead asked:
#   1. E355 checks (selftest, kill rows, smokes), CPU, 2 threads, nice 19, one process;
#   2. +3 min (the E354 post jobs and E357's gate take their GPU tickets first): E353 seed 1 (cp_s1 / n1_s1) as the manual start of day29's seed-1 part, + the watchdog that stops them at 01:30 (500-step checkpoint) for E357's tickets;
#   3. when the checks ended with exit 0: the 9 E355 arms (start_arms_e355.txt, serial --after chain, s20 first = the kill row).
# (E352 step 0 starts by itself: temp/day33/day33.sh waits for the same event.)  Log: temp/_session_scratch_s292/night_go.log
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
S=temp/_session_scratch_s292
LOG=$S/night_go.log
say() { echo "$(date +%H:%M:%S) $*" >> $LOG; }
rm -f $S/check_e355.rc
say "night_go waits for the end of jrecur_train.py"
until ! pgrep -f "^python jrecur_train.py" >/dev/null; do sleep 30; done
say "E354 trainings through"
sleep 60
( nice -n 19 bash temp/e346/check_e355.sh >> temp/e346/out/check_stdout.log 2>&1; echo $? > $S/check_e355.rc ) &
say "check_e355 started"
sleep 120
env DAY29_SKIP_WAIT=1 DAY29_STOP_AFTER_S1=1 nohup bash temp/day29/day29.sh >> temp/day29/day29_stdout.log 2>&1 &
nohup bash $S/e353_s1_watchdog.sh 0130 >> $S/e353_s1_watchdog.log 2>&1 &
say "E353 seed 1 started (day29 seed-1 part), watchdog 01:30"
while [ ! -f $S/check_e355.rc ]; do sleep 30; done
rc=$(cat $S/check_e355.rc)
say "check_e355 exit $rc"
if [ "$rc" = "0" ]; then
  grep -E '^env ' temp/e346/start_arms_e355.txt | while IFS= read -r line; do
    eval "$line" >> temp/e346/out/arms_launch.log 2>&1 &
    sleep 3
  done
  say "E355 arms launched (serial --after chain)"
else
  say "check_e355 FAILED: the arms are NOT launched; read temp/e346/out/check_e355.log"
fi
wait
