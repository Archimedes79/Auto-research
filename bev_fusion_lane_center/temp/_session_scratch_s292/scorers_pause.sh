#!/bin/bash
# usage: scorers_pause.sh stop|cont  -> SIGSTOP / SIGCONT of my CPU card scorers (E291 cards: `python jends_eval.py score --run e291_*`, E353 cards: `--run e353_*`) for a quiet-host timing block
sig=STOP; [ "$1" = "cont" ] && sig=CONT
for p in $(pgrep -f "^python jends_eval.py score --run e(291|353)_" ; pgrep -f "^python jends_eval.py score --run e291_" ); do
  kill -$sig "$p" 2>/dev/null && echo "$sig $p $(ps -o etime=,args= -p $p | cut -c1-90)"
done
# pgrep -f uses extended regex: the first pattern alone covers both; duplicates are harmless
echo done
