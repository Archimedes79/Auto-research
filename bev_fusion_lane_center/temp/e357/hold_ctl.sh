#!/bin/bash
# E357: keep the two long CPU jobs of the chain (jtok_eval.py ctl, 36 min each) from running next to E354's host-bound trainings
# (the lane lead's request of 2026-11-15 21:51). While a jrecur_train.py process runs, every `jtok_eval.py ctl --lattice 2.5` python
# process of my own chain is SIGSTOPped as soon as it appears; when E354's trainings are through, the stopped ones get SIGCONT and this
# watcher ends. It touches only processes of the E357 chain; the chain script itself is not edited (it simply waits on its child).
cd "$(dirname "$0")/../.."
L=temp/e357/hold_ctl.log
echo "$(date '+%H:%M:%S') watcher starts" >> $L
STOPPED=""
training() { ps -eo cmd | grep -E "[p]ython jrecur_train.py" | grep -q .; }
while training; do
  for p in $(ps -eo pid,cmd | grep -E "[p]ython .*jtok_eval.py ctl --lattice 2.5" | awk '{print $1}'); do
    case " $STOPPED " in *" $p "*) ;; *) kill -STOP $p && STOPPED="$STOPPED $p" && echo "$(date '+%H:%M:%S') stopped $p" >> $L;; esac
  done
  sleep 15
done
for p in $STOPPED; do kill -CONT $p 2>/dev/null && echo "$(date '+%H:%M:%S') continued $p" >> $L; done
echo "$(date '+%H:%M:%S') E354 trainings are through, watcher ends" >> $L
