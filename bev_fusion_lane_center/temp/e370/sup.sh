#!/bin/bash
# E370 training supervisor for ONE run (resume on a crash). usage: bash temp/e370/sup.sh <run id suffix> <seed> <timeout_min> <name> [jtoken_train.py flags...]
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
ID=${1:?}; S=${2:?}; TO=${3:?}; NAME=${4:?}; shift 4
L=temp/e370/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup $ID] $*" >> $L; }
source temp/e370/texts.sh
exec 8>/tmp/e370_sup_$ID.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts: $NAME $*"
n=0
while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
  if pgrep -f "run.py --id E370.$ID " > /dev/null; then sleep 20; continue; fi
  n=$((n+1)); [ $n -gt 12 ] && { say "12 launches, giving up"; exit 1; }
  say "launch $n"
  python3 ../autoresearch/tools/run.py --id E370.$ID --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --ident e292_a --bs 8 --lr 1e-3 --workers 4 "$@" >> temp/e370/train_$ID.out 2>&1
  sleep 15
done
say "training done"
