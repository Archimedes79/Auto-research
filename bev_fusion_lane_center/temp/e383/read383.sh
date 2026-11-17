#!/bin/bash
# E383 read-out of one head (the E378 pattern). usage: bash temp/e383/read383.sh dump|cpu <arm>_s<S>      (arm f or k)
#   dump: VAL + TRAIN-sample (every 3) dumps on the GPU (blocking)
#   cpu : the per-class TRAIN sweep -> the pinned cell (choose378.py, logged BEFORE any VAL score) ->
#         f: the VAL score at D115 against the E373 champion head of the same cache (--ref-run: the same K 512 tokens: the PARITY card), then its own pinned cell (reported, with --ref-run)
#         k: the VAL score at its OWN pinned cell (the deciding read) and at D115 (reported). NO --ref-run: the K 384 tokens are not the K 512 tokens (jtoken_eval.same_tokens refuses); k is compared with the stored
#            champion numbers (runs/e373_cb_s<S>/token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json) and with f's card by hand, pooled VAL numbers (the rule has no interval clause)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
MODE=${1:?mode}; A=${2:?arm_s}; ARM=${A%%_*}; S=${A#*_s}; R=e383_$A
source temp/e383/texts383.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read $A $MODE] $*" >> temp/e383/chain383.log; }
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
if [ "$MODE" = dump ]; then
  $RUN --id E383.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3"
  say "exit $?"
elif [ "$MODE" = cpu ]; then
  $RUN --id E383.sweep_$A --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3 --grid perclass" || { say "sweep failed"; exit 1; }
  CELL=$(python3 temp/e378/choose378.py $R) || { say "chooser failed"; exit 1; }
  say "pinned cell $CELL (from the TRAIN table, before any VAL score)"
  IFS=, read LT LTL KT G <<< "$CELL"
  GATE=$([ "$G" = off ] && echo --no-none-gate)
  REF=""; [ "$ARM" = f ] && REF="--ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off"
  if [ "$ARM" = k ]; then
    $RUN --id E383.cell_$A --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S --link-thr $LT --link-thr-long $LTL --keep-thr $KT $GATE"
    say "own cell $CELL (the deciding read of k) exit $?"
  fi
  $RUN --id E383.d115_$A --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate $REF"
  say "d115 (f: the PARITY card against the champion head; k: reported) exit $?"
  if [ "$ARM" = f ] && [ "$CELL" != "0.3,0.2,0.6,off" ]; then
    $RUN --id E383.cell_$A --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S --link-thr $LT --link-thr-long $LTL --keep-thr $KT $GATE $REF"
    say "own cell $CELL (reported) exit $?"
  fi
else
  say "unknown mode $MODE"; exit 1
fi
