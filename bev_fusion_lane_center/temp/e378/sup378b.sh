#!/bin/bash
# E378: x_s1 only (the first supervisor was stopped at 17:52 on the lead's word: no GPU training between 17:30 and 22:00). Waits for 22:00, then trains
# x_s1 (cache 1, head 5) on the GPU queue, relaunching after a crash (resume), then its dumps (blocking) and its CPU read-out (blocking).
# usage: bash temp/e378/sup378b.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e378/chain378.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup378b] $*" >> $L; }
source temp/e378/texts378.sh
exec 8>/tmp/e378b_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "waiting for the night window (22:00)"
until [ "$(date +%H%M)" -ge 2200 ] || [ "$(date +%H%M)" -lt 0500 ]; do sleep 30; done
say "night window open"
S=1; A=x_s1; H=5; NAME=e378_$A; n=0
while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
  if pgrep -f "run.py --id E378.$A " > /dev/null; then sleep 30; continue; fi
  if grep -qiE "loss (nan|inf)|not finite" runs/$NAME/log.txt temp/e378/train_$A.out 2>/dev/null; then say "$A: a non-finite loss (the card's kill): no relaunch"; exit 1; fi
  n=$((n+1)); [ $n -gt 12 ] && { say "$A: 12 launches, giving up"; exit 1; }
  say "$A launch $n (cache seed $S, head seed $H, --none-weight 0.25 --pos-loss max)"
  python3 ../autoresearch/tools/run.py --id E378.$A --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --pos-loss max >> temp/e378/train_$A.out 2>&1
  sleep 20
done
say "$A training done"
bash temp/e378/read378.sh dump $A >> temp/e378/read_$A.out 2>&1
bash temp/e378/read378.sh cpu $A >> temp/e378/read_$A.out 2>&1
say "$A read-out done"
