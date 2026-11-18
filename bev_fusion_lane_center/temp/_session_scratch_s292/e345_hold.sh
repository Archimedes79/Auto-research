#!/bin/bash
# usage: e345_hold.sh stop|cont -> SIGSTOP / SIGCONT of my E345 day21 jobs that would load the host or hold the GPU in the timing block:
# the run.py launchers of E345.v_e324s_s0 / s1, their jprune_probe children, and the E345 CPU scorers (jprune_table.py score)
sig=STOP; [ "$1" = "cont" ] && sig=CONT
for p in $(pgrep -f "run.py --id E345.v_e324s_s"); do kill -$sig "$p" && echo "$sig launcher $p"; done
for p in $(pgrep -f "python jprune_probe.py"); do kill -$sig "$p" && echo "$sig probe $p"; done
for p in $(pgrep -f "^python jprune_table.py"); do kill -$sig "$p" && echo "$sig table $p"; done
echo done
