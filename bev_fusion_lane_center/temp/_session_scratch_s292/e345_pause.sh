#!/bin/bash
# usage: e345_pause.sh stop|cont  -> SIGSTOP / SIGCONT of my two E345 scoring python processes (jprune_table.py score, run ids E345.s_e324s_s0 / s1)
sig=STOP; [ "$1" = "cont" ] && sig=CONT
for p in $(pgrep -f "^python jprune_table.py score --run e324_s_s"); do
  kill -$sig "$p" && echo "$sig $p $(ps -o etime=,stat= -p $p)"
done
