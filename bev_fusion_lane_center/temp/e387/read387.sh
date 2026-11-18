#!/bin/bash
# E387 read-out of one f arm (card section 0). usage: bash temp/e387/read387.sh dump|cpu f_s<S>
#   dump: the GPU stream dumps of the arm (VAL, TRAIN 873 frames, VAL in the mode `off`) through run.py's GPU queue
#   cpu : every3 -> the per-class TRAIN sweep -> the arm's OWN pinned cell (logged BEFORE any VAL score) -> the VAL scores in the stream mode, f against z at z's own cell, f against r at r's own cell, f at the COMMON cell against z at it, the D115 pair, f against its own `off` dump (the use clause) -> the existence AP
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
MODE=${1:?mode}; A=${2:?arm_s}; S=${A#*_s}; R=e387_$A
source temp/e387/texts387.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e387 read $A $MODE] $*" >> temp/e387/chain.log; }
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
cellof() { grep "^$1 " temp/e387/cells.log | tail -1 | cut -d' ' -f2; }
if [ "$MODE" = dump ]; then
  $RUN --id E387.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jpast_eval.py dump --run $R --seed $S --split val && $PYR jpast_eval.py dump --run $R --seed $S --split train && $PYR jpast_eval.py dump --run $R --seed $S --split val --mode off"
  say "exit $?"
elif [ "$MODE" = cpu ]; then
  $PY token_readout.py every3 $R || { say "every3 failed"; exit 1; }
  $RUN --id E387.sweep_$A --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3 --grid perclass --jobs 3" || { say "sweep failed"; exit 1; }
  CELL=$($PY jtoken_eval.py choose-cell --runs $R | tail -1) || { say "chooser failed"; exit 1; }
  [[ "$CELL" =~ ^[0-9.]+,[0-9.]+,[0-9.]+,(on|off)$ ]] || { say "chooser failed: $CELL"; exit 1; }
  say "pinned cell $CELL (own TRAIN table, before any VAL score)"
  echo "$A $CELL" >> temp/e387/cells.log
  ZC=$(cellof z_s$S); RC=$(cellof r_s$S); CC=$(cellof common)
  [ -n "$ZC" ] && [ -n "$RC" ] && [ -n "$CC" ] || { say "cells of z / r / common missing in temp/e387/cells.log"; exit 1; }
  flags() { IFS=, read LT LTL KT G <<< "$1"; echo "--link-thr $LT --link-thr-long $LTL --keep-thr $KT $([ "$G" = off ] && echo --no-none-gate)"; }
  score() {  # ID CELL REFRUN REFCELL
    $RUN --id E387.$1 --seed $S --retries 0 --require-card --split val --timeout 150 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S $(flags $2) --ref-run $3 --ref-run-cell $4"
    say "$1 exit $?"
  }
  score own_z_$A "$CELL" e368_z_s$S "$ZC" &
  score own_r_$A "$CELL" e368_r_s$S "$RC" &
  score common_z_$A "$CC" e368_z_s$S "$CC" &
  wait
  score d115_z_$A "0.3,0.2,0.6,off" e368_z_s$S "0.3,0.2,0.6,off" &
  score off_$A "$CELL" ${R}_off "$CELL" &
  $PY token_readout.py ap $R e368_z_s$S e368_r_s$S >> temp/e387/ap.out 2>&1 &
  wait
  say "cpu read-out done"
else
  say "unknown mode $MODE"; exit 1
fi
