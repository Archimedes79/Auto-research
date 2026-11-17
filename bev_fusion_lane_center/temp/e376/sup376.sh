#!/bin/bash
# E376 (card v2) training supervisor for ONE run on one GPU slot: keeps run.py job E376.<id> alive until the training is done (jtoken_train resumes from
# the 500-step checkpoint; the E362 lesson), then the GPU dumps on the same slot (blocking), then the CPU read-out in the background.
# usage: bash temp/e376/sup376.sh mx_s<S> <timeout_min>     (the card: 12000 x 0.26 s x 1.3 x 1.3 = 88 min; head seed = S + 2 as E373)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
A=${1:?id}; TO=${2:?timeout}
ARM=${A%_s*}; S=${A#*_s}; H=$((S+2)); NAME=e376_$A
[ "$ARM" = mx ] || { echo "unknown arm $ARM"; exit 1; }
L=temp/e376/chain376.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup $A] $*" >> $L; }
source temp/e376/texts376.sh
exec 8>/tmp/e376_sup_$A.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts, timeout $TO min (cache seed $S, head seed $H, --pos-loss max)"
n=0
while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
  if pgrep -f "run.py --id E376.$A " > /dev/null; then sleep 30; continue; fi
  n=$((n+1)); [ $n -gt 12 ] && { say "12 launches, giving up"; exit 1; }
  say "launch $n"
  python3 ../autoresearch/tools/run.py --id E376.$A --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --pos-loss max >> temp/e376/train_$A.out 2>&1
  sleep 20
done
say "training done"
bash temp/e376/read376.sh dump $A >> temp/e376/read_$A.out 2>&1
bash temp/e376/read376.sh cpu $A >> temp/e376/read_$A.out 2>&1
say "read-out (a) + sweep done"
