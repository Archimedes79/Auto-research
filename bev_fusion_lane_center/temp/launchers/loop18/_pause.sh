#!/bin/bash
# usage: _pause.sh STOP|CONT <pattern in the command line>  (SIGSTOP / SIGCONT of the python processes of one of MY runs; the GPU memory stays allocated)
sig=$1; pat=$2
for p in $(ps -eo pid,args | grep "python p16_train.py\|python p18_\|python p17_" | grep -F -- "$pat" | grep -v grep | awk '{print $1}'); do kill -$sig $p 2>/dev/null; echo "$sig $p"; done
