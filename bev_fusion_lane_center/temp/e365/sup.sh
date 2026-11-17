#!/bin/bash
# E365 training supervisor for ONE seed: waits for the TRAIN cache of B' seed S, then keeps a run.py job E365.s1_s<S> alive until the training is done
# (run.py resumes a failed id from the 500-step checkpoint; the E362 lesson: the GPU crashes now and then). usage: bash temp/e365/sup.sh <seed> <timeout_min>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; TO=${2:?timeout minutes}
L=temp/e365/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [sup s$S] $*" >> $L; }
source temp/e365/texts.sh
exec 8>/tmp/e365_sup_s$S.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts, timeout $TO min"
until ls runs/e365_cache/be292_a_${S}_K*/train/manifest.json > /dev/null 2>&1; do sleep 30; done
say "train cache of seed $S is ready"
n=0
while ! grep -q "^[0-9:]* done e365_s1_s$S\$" runs/e365_s1_s$S/log.txt 2>/dev/null; do
  if pgrep -f "run.py --id E365.s1_s$S " > /dev/null; then sleep 30; continue; fi
  n=$((n+1)); [ $n -gt 12 ] && { say "12 launches, giving up"; exit 1; }
  say "launch $n"
  $RUN --id E365.s1_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name e365_s1_s$S --seed $S --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 >> temp/e365/train_s$S.out 2>&1
  echo $? > temp/e365/train_s$S.rc
  sleep 20
done
say "training done"
