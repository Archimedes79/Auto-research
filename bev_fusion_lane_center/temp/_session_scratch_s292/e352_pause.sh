#!/bin/bash
# usage: e352_pause.sh stop|cont -> SIGSTOP / SIGCONT of the E352 step-0 chain's worker processes (jprune_b.py dump / check, jends_eval.py score --run e352*); the run.py launchers and day30.sh keep running (the 120-min job timeout counts the pause: harmless for pauses up to ~100 min per job, see the coder's notes)
sig=STOP; [ "$1" = "cont" ] && sig=CONT
for p in $(pgrep -f "^python jprune_b.py"; pgrep -f "^python jends_eval.py score --run e352"); do
  kill -$sig "$p" 2>/dev/null && echo "$sig $p $(ps -o etime=,args= -p $p | cut -c1-100)"
done
echo done
