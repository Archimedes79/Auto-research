#!/bin/bash
# E372: the live dumps and read-outs of the arms. sup372.sh trains the arms correctly but its done-check pattern ("^time done NAME$") does not match
# jlive_train's done line ("time done NAME: model.pt holds ..."), so after each training it spins 8 refused relaunches and SKIPS the arm's dump and read-out.
# This script (started 00:55, sup372.sh untouched while it runs) waits for each arm's done line and does the dump (GPU, run.py) and the read-out (CPU).
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e372/chain372.log
say() { echo "$(date '+%m-%d %H:%M:%S') [dumps372] $*" >> $L; }
source temp/e372/texts372.sh
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
D115="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
exec 8>/tmp/e372_dumps.lock
flock -n 8 || { say "already runs"; exit 0; }
say "starts"
for A in z_s0 s4_s0 b_s0 z_s1 s4_s1 b_s1; do
  ARM=${A%_s*}; S=${A#*_s}; NAME=e372_$A
  until grep -qE "^[0-9:]+ done $NAME(:|\$)" runs/$NAME/log.txt 2>/dev/null || [ -f runs/$NAME/KILLED ]; do sleep 30; done
  if [ -f runs/$NAME/KILLED ]; then say "$A KILLED: no dump / read-out"; continue; fi
  if [ ! -f runs/$NAME/token_val_live.pkl ]; then
    $RUN --id E372.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --live --run $NAME --seed $S --split val" >> temp/e372/read_$A.out 2>&1
    say "$A live dump exit $?"
  fi
  EXTRA=""; [ "$ARM" != z ] && EXTRA=" && $PY jtoken_eval.py score --live --run $NAME --seed $S $D115 --ref-run e372_z_s$S && $PY jtoken_eval.py live-diff --run $NAME --seed $S --against e373_cb_s$S"
  ( if [ "$ARM" != z ]; then until ls runs/e372_z_s$S/token_val_live.pkl > /dev/null 2>&1 || [ -f runs/e372_z_s$S/KILLED ]; do sleep 60; done; fi
    $RUN --id E372.read_$A --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --live --run $NAME --seed $S $D115 --ref-run e373_cb_s$S$EXTRA" >> temp/e372/read_$A.out 2>&1
    say "read_$A exit $?" ) &
done
wait
say "all dumps and read-outs done"
