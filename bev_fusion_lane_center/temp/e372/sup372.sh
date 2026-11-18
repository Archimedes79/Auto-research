#!/bin/bash
# E372 (card v3.2) night supervisor on ONE GPU slot: (1) the parity dumps (the champion heads e373_cb_s0 / s1 read LIVE with the untouched B', bs 4) and their parity scores
# against their cache read-outs; (2) the arms z_s0, s4_s0, b_s0, z_s1, s4_s1, b_s1 (3000 steps, timeout 90, resume after a crash, a KILLED folder never resumed: that arm is
# skipped and reported), each followed by its live VAL dump on the same slot and its CPU read-out in the background. usage: bash temp/e372/sup372.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e372/chain372.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup372] $*" >> $L; }
source temp/e372/texts372.sh
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
D115="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
exec 8>/tmp/e372_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts"
# (1) parity
for S in 0 1; do
  if ! ls runs/e373_cb_s$S/token_val_live.pkl > /dev/null 2>&1; then
    $RUN --id E372.parity_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --live --run e373_cb_s$S --seed $S --split val" >> temp/e372/parity_s$S.out 2>&1
    say "parity_s$S dump exit $?"
  fi
  ( $RUN --id E372.parity_read_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --live --run e373_cb_s$S --seed $S $D115 --ref-run e373_cb_s$S --ref-cache" >> temp/e372/parity_s$S.out 2>&1; say "parity_read_s$S exit $?" ) &
done
# (2) arms
for A in z_s0 s4_s0 b_s0 z_s1 s4_s1 b_s1; do
  ARM=${A%_s*}; S=${A#*_s}; NAME=e372_$A; n=0
  case $ARM in z) X="--train-nodes 0";; b) X="--train-nodes 1 --live-inputs 0";; s4) X="--train-nodes 1 --live-inputs 1";; esac
  while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
    if [ -f runs/$NAME/KILLED ]; then say "$A KILLED ($(cat runs/$NAME/KILLED | head -c 200)): not resumed, next arm"; continue 2; fi
    if pgrep -f "run.py --id E372.$A " > /dev/null; then sleep 30; continue; fi
    n=$((n+1)); [ $n -gt 8 ] && { say "$A: 8 launches, giving up, next arm"; continue 2; }
    say "$A launch $n ($X)"
    $RUN --id E372.$A --seed $S --gpu --ram-mb 8000 --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jlive_train.py --name $NAME --seed $S --head-run e373_cb_s$S $X --steps 3000 --bs 8 --lr 1e-4 --workers 4 >> temp/e372/train_$A.out 2>&1
    sleep 20
  done
  say "$A training done"
  $RUN --id E372.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --live --run $NAME --seed $S --split val" >> temp/e372/read_$A.out 2>&1
  say "$A live dump exit $?"
  EXTRA=""; [ "$ARM" != z ] && EXTRA=" && $PY jtoken_eval.py score --live --run $NAME --seed $S $D115 --ref-run e372_z_s$S && $PY jtoken_eval.py live-diff --run $NAME --seed $S --against e373_cb_s$S"
  ( if [ "$ARM" != z ]; then until grep -q "^[0-9:]* done e372_z_s$S\$" runs/e372_z_s$S/log.txt 2>/dev/null && ls runs/e372_z_s$S/token_val_live.pkl > /dev/null 2>&1 || [ -f runs/e372_z_s$S/KILLED ]; do sleep 60; done; fi
    $RUN --id E372.read_$A --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --live --run $NAME --seed $S $D115 --ref-run e373_cb_s$S$EXTRA" >> temp/e372/read_$A.out 2>&1
    say "read_$A exit $?" ) &
done
wait
say "all arms and read-outs done"
