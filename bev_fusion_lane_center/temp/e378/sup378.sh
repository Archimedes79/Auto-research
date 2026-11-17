#!/bin/bash
# E378 (card v1+) supervisor: ONE job family on slot 2: trains x_s0 (cache 0, head 4) then x_s1 (cache 1, head 5), --pos-loss max on the champion recipe,
# relaunching a run.py id after a crash (resume), each followed by its GPU dumps on the same slot (blocking) and its CPU read-out in the background.
# usage: bash temp/e378/sup378.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e378/chain378.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup378] $*" >> $L; }
source temp/e378/texts378.sh
exec 8>/tmp/e378_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts"
for S in 0 1; do
  A=x_s$S; H=$((S+4)); NAME=e378_$A; n=0
  while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
    if pgrep -f "run.py --id E378.$A " > /dev/null; then sleep 30; continue; fi
    n=$((n+1)); [ $n -gt 12 ] && { say "$A: 12 launches, giving up"; exit 1; }
    say "$A launch $n (cache seed $S, head seed $H, --none-weight 0.25 --pos-loss max)"
    python3 ../autoresearch/tools/run.py --id E378.$A --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 55 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --pos-loss max >> temp/e378/train_$A.out 2>&1
    sleep 20
  done
  say "$A training done"
  bash temp/e378/read378.sh dump $A >> temp/e378/read_$A.out 2>&1
  bash temp/e378/read378.sh cpu $A >> temp/e378/read_$A.out 2>&1 &
done
wait
say "both trainings and read-outs done"
