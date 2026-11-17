#!/bin/bash
# E367 training supervisor for ONE run: keeps a run.py job E367.<id> alive until the training is done (run.py resumes a failed id from the 500-step checkpoint;
# the E362 lesson: the GPU crashes now and then). usage: bash temp/e367/sup.sh <id> <seed> <timeout_min> <steps> [extra jtoken_train.py flags...]   e.g.  bash temp/e367/sup.sh b5_s0 0 130 12000 --none-weight 0.5
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
ID=${1:?id}; S=${2:?seed}; TO=${3:?timeout}; STEPS=${4:?steps}; shift 4
L=temp/e367/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [sup $ID] $*" >> $L; }
source temp/e367/texts.sh
NAME=e367_$ID
exec 8>/tmp/e367_sup_$ID.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts, timeout $TO min, steps $STEPS, flags: $*"
n=0
while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
  if pgrep -f "run.py --id E367.$ID " > /dev/null; then sleep 30; continue; fi
  n=$((n+1)); [ $n -gt 12 ] && { say "12 launches, giving up"; exit 1; }
  say "launch $n"
  $RUN --id E367.$ID --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --ident e292_a --steps $STEPS --bs 8 --lr 1e-3 --workers 4 "$@" >> temp/e367/train_$ID.out 2>&1
  sleep 20
done
say "training done"
