#!/bin/bash
# E353 seed-1 trainings must release both GPU slots by 01:30 (T0 / E357 tickets): at that time stop what still runs at its last 500-step checkpoint (the trainer resumes ckpt.pt: restart `day29.sh` later)
# usage: e353_s1_watchdog.sh [HHMM]  (default 0130)
until=${1:-0130}
while true; do
  now=$(date +%H%M)
  # wait through midnight: stop only when the hour is before 12:00 and the time is >= until
  if [ "$now" -ge "$until" ] && [ "$now" -lt 1200 ]; then break; fi
  sleep 30
done
echo "watchdog fires $(date +%H:%M:%S)"
for p in $(pgrep -f "^python jnarrow_train.py --name e353_(cp|n1)_s1"); do
  kill -TERM "$p" 2>/dev/null && echo "TERM $p"
done
echo done
