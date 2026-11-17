#!/bin/bash
# E376 training supervisor: ONE job family on one GPU slot. Trains the four runs one after another (lw_s0, fl_s0, lw_s1, fl_s1: both arms get
# their seed-0 head first) and relaunches a run.py id after a crash (jtoken_train resumes from the 500-step checkpoint; the E362 lesson).
# usage: bash temp/e376/sup376.sh <timeout_min>     (the card: steps x 0.26 s x 1.3 x 1.3 = 88 min for 12000 steps)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
TO=${1:?timeout}
L=temp/e376/chain376.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup376] $*" >> $L; }
source temp/e376/texts376.sh
exec 8>/tmp/e376_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts, timeout $TO min"
for A in lw_s0 fl_s0 lw_s1 fl_s1; do
  ARM=${A%_s*}; S=${A#*_s}; H=$((S+2)); NAME=e376_$A
  case $ARM in lw) X="--none-weight-long 1.0";; fl) X="--soft-f-long 1.0";; *) say "unknown arm $ARM"; exit 1;; esac
  n=0
  while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
    if pgrep -f "run.py --id E376.$A " > /dev/null; then sleep 30; continue; fi
    n=$((n+1)); [ $n -gt 12 ] && { say "$A: 12 launches, giving up"; exit 1; }
    say "$A launch $n (cache seed $S, head seed $H, $X)"
    python3 ../autoresearch/tools/run.py --id E376.$A --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 $X >> temp/e376/train_$A.out 2>&1
    sleep 20
  done
  say "$A training done"
  bash temp/e376/read376.sh dump $A >> temp/e376/read_$A.out 2>&1            # GPU dumps on THIS slot, before the next training
  bash temp/e376/read376.sh cpu $A >> temp/e376/read_$A.out 2>&1 &           # CPU scores in the background
done
wait
say "all four trainings done"
