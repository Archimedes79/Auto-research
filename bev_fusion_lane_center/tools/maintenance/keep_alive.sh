#!/bin/bash
# keep_alive.sh - the shared part of the night / chain supervisors (promoted 2026-11-17 from the supervisors temp/e367/sup.sh, temp/e368/sup.sh, temp/e374/sup.sh, temp/e378/sup378b.sh, which repeated it):
# a supervisor script SOURCES this file and keeps `run.py` jobs alive until their training says 'done' (the trainer resumes from its 500-step checkpoint, the GPU crashes now and then).
#
#     cd "$(dirname "$0")/../.."                                  # the project folder (the log paths are relative to it)
#     source tools/maintenance/keep_alive.sh
#     ka_init temp/eNNN/chain.log eNNN lane_eNNN_sup              # log file, tag in the log lines, lock name (/tmp/<lock>.lock): a second copy of the supervisor says so and exits 0
#     ka_window 2200 0700                                         # optional: wait until the clock is inside the window (the GPU nights 22:00 - 07:00; a start after the end is a window over midnight)
#     ka_keep E374.m_s0 e374_m_s0 6 python3 ../autoresearch/tools/run.py --id E374.m_s0 ... -- bash tools/env/_pyr.sh jmap_train.py --name e374_m_s0 ...
#
# ka_keep RUN_ID NAME MAX CMD...: runs CMD (one run.py launch, in the foreground) again and again until runs/NAME/log.txt holds a line '<time> done NAME' (the trainer's last line), at most MAX launches (then it
# returns 1); while a `run.py --id RUN_ID ` of another supervisor is alive it waits instead of launching; nothing is launched when the log already says done. Environment: KA_RUNS (default runs), KA_POLL (seconds between
# looks at a live job, default 30), KA_SLEEP (seconds after a launch ended, default 20), KA_NOW (HHMM for the tests). Test: tools/maintenance/test_keep_alive.sh.
KA_RUNS=${KA_RUNS:-runs}
KA_POLL=${KA_POLL:-30}
KA_SLEEP=${KA_SLEEP:-20}

say() { echo "$(date '+%m-%d %H:%M:%S') [$KA_TAG] $*" >> "$KA_LOG"; }

ka_init() {  # LOG TAG LOCK
  KA_LOG=$1; KA_TAG=$2
  mkdir -p "$(dirname "$KA_LOG")"
  exec 8>"/tmp/$3.lock"
  flock -n 8 || { say "supervisor already runs"; exit 0; }
}

ka_in_window() {  # START END (HHMM): the clock is in [START, END), over midnight when START > END
  local now=${KA_NOW:-$(date +%H%M)} s=$((10#$1)) e=$((10#$2))
  now=$((10#$now))
  if [ "$s" -le "$e" ]; then [ "$now" -ge "$s" ] && [ "$now" -lt "$e" ]; else [ "$now" -ge "$s" ] || [ "$now" -lt "$e" ]; fi
}

ka_window() {  # START END: wait for the window
  say "waiting for the window $1 - $2"
  until ka_in_window "$1" "$2"; do sleep "${KA_POLL}"; done
  say "window open"
}

ka_keep() {  # RUN_ID NAME MAX CMD...
  local id=$1 name=$2 max=$3 n=0
  shift 3
  while ! grep -q "^[0-9:]* done $name\$" "$KA_RUNS/$name/log.txt" 2>/dev/null; do
    if pgrep -f "run.py --id $id " > /dev/null; then sleep "$KA_POLL"; continue; fi
    n=$((n + 1))
    [ "$n" -gt "$max" ] && { say "$id: $max launches, giving up"; return 1; }
    say "$id launch $n"
    "$@"
    sleep "$KA_SLEEP"
  done
  say "$id done"
}
