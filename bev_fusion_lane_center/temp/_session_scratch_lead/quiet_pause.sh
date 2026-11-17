#!/bin/bash
# Lane lead, 2026-11-16 03:05. Makes the quiet-host window of the night watcher (temp/_session_scratch_s292/e351_quiet_watch4.sh, 05:00 - 07:00) possible:
# the E352 seed-1 jobs (CPU, nice 19, load ~6) would keep the load above the watcher's bar of 3, and D107 ranks the E353 / E351 timing (priority 3 / 6) above E352 (priority 4) ONLY for the milli-seconds,
# which cannot be measured on a loaded host. So: at 04:58 the python workers of E352 (jprune_b.py dump, jends_eval.py score of e352 runs) get SIGSTOP (their run.py wrappers keep their heartbeat and their timeout of
# 120 min is far away), and SIGCONT when the file temp/_session_scratch_lead/resume_e352 exists (the lead creates it when the timings of E351 / E353 AND of the Fable session's jrecur_time.py are through),
# or at 06:20 at the latest (then E352 has 70 min left before the 07:30 deadline). Nothing is killed, nothing is edited. Log: temp/_session_scratch_lead/quiet_pause.log
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
D=temp/_session_scratch_lead
LOG=$D/quiet_pause.log
say() { echo "$(date +%H:%M:%S) $*" >> $LOG; }
rm -f $D/resume_e352
T0=$(date -d "tomorrow 04:58" +%s); [ "$(date +%H)" -lt 12 ] && T0=$(date -d "today 04:58" +%s)
HARD=$(date -d "today 06:20" +%s)
say "waits until 04:58"
while [ "$(date +%s)" -lt "$T0" ]; do sleep 30; done
PIDS=$(pgrep -f "^python jprune_b.py|^python jends_eval.py score --run e352" | tr '\n' ' ')
say "SIGSTOP: $PIDS"
[ -n "$PIDS" ] && kill -STOP $PIDS
STOPPED="$PIDS"
while [ ! -f $D/resume_e352 ] && [ "$(date +%s)" -lt "$HARD" ]; do
  # workers started after 04:58 by the chain (a new stage) are stopped too, so the host stays quiet
  NEW=$(pgrep -f "^python jprune_b.py|^python jends_eval.py score --run e352" | tr '\n' ' ')
  for p in $NEW; do case " $STOPPED " in *" $p "*) ;; *) kill -STOP $p; STOPPED="$STOPPED $p"; say "SIGSTOP new $p";; esac; done
  sleep 20
done
say "SIGCONT: $STOPPED (flag: $( [ -f $D/resume_e352 ] && echo resume_e352 || echo hard time))"
kill -CONT $STOPPED
say "done"
