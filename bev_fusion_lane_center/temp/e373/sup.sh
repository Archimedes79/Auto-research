#!/bin/bash
# E371 training supervisor for one fresh head: keeps a run.py job alive until the training is done (resume from the 500-step checkpoint).
# usage: bash temp/e373/sup.sh <cache seed> <head seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; H=${2:?head seed}; NAME=e373_cb_s$S
L=temp/e373/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup cb_s$S] $*" >> $L; }
source temp/e373/texts.sh
exec 8>/tmp/e373_sup_$S.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts (cache seed $S, head seed $H)"
n=0
while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
  if pgrep -f "run.py --id E373.cb_s$S " > /dev/null; then sleep 30; continue; fi
  n=$((n+1)); [ $n -gt 12 ] && { say "12 launches, giving up"; exit 1; }
  say "launch $n"
  python3 ../autoresearch/tools/run.py --id E373.cb_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 130 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --none-weight 0.25 --steps 12000 --bs 8 --lr 1e-3 --workers 4 >> temp/e373/train_s$S.out 2>&1
  sleep 20
done
say "training done"
