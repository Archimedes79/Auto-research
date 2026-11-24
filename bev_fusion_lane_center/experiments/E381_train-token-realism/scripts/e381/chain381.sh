#!/bin/bash
# E381 card v2.1 (D126) chain: (1) the two noisy heads in parallel (GPU, 2 slots), (2) the kill checks from their logs, (3) per head the VAL + TRAIN-sample cache dumps (GPU), (4) the per-class TRAIN sweeps (CPU),
# (5) the pinned cells (logged BEFORE any VAL score; the recipe cell when they differ), (6) the VAL reads against the E373 heads at D115 (+ D115 and own-cell reads, anatomy).
# usage: bash temp/e381/chain381.sh   (from the problem folder, in WSL)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e381/chain381.log
say() { echo "$(date '+%m-%d %H:%M:%S') [chain381] $*" >> $L; }
source temp/e381/texts381.sh
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
PYR="bash tools/env/_pyr.sh"
declare -A SPV=([0]=0.184 [1]=0.181) SEV=([0]=0.621 [1]=0.601) HS=([0]=2 [1]=3)
exec 8>/tmp/e381_chain.lock
flock -n 8 || { say "already runs"; exit 0; }
say "starts"
H=$(date +%H%M); if [ "$H" -ge 0500 ] && [ "$H" -lt 2200 ]; then say "after the 05:00 start limit: nothing started"; exit 1; fi
# (1) training, both seeds in parallel
for S in 0 1; do
  ( $RUN --id E381.n_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- \
      $PYR jtoken_train.py --name e381_n_s$S --seed $S --head-seed ${HS[$S]} --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --noise-pos ${SPV[$S]} --noise-e ${SEV[$S]} \
      >> temp/e381/n_s$S.out 2>&1; say "n_s$S exit $?" ) &
  sleep 20
done
wait
# (2) the kill checks
KILL=0
for S in 0 1; do
  $PY temp/e381/kill381.py $S >> temp/e381/kill_s$S.out 2>&1; k=$?; say "kill check s$S exit $k: $(tail -1 temp/e381/kill_s$S.out)"; [ $k -eq 0 ] || KILL=1
done
[ $KILL -eq 0 ] || { say "a kill fired: INVALID, no read-out (card v2.1)"; exit 2; }
# (3) dumps (GPU), (4) sweeps (CPU, both in parallel after the dumps)
for S in 0 1; do
  $RUN --id E381.dump_n_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 30 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run e381_n_s$S --seed $S --split val && $PYR jtoken_eval.py dump --run e381_n_s$S --seed $S --split train --every 3" >> temp/e381/dump_s$S.out 2>&1
  say "dump_n_s$S exit $?"
done
for S in 0 1; do
  ( $RUN --id E381.sweep_n_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py sweep --run e381_n_s$S --seed $S --split train --every 3 --grid perclass" >> temp/e381/sweep_s$S.out 2>&1; say "sweep_n_s$S exit $?" ) &
done
wait
# (5) the pinned cells, logged before any VAL score
declare -A CELL
for S in 0 1; do
  CELL[$S]=$($PY jtoken_eval.py choose-cell --runs e381_n_s$S --split train --every 3 2>>temp/e381/cell.err | tail -1)
  say "pinned (b) cell of e381_n_s$S (its own TRAIN table): ${CELL[$S]}"
  python3 ../autoresearch/tools/ledger.py note E381.n_s$S --notes "pinned (b) cell from its own TRAIN table (choose-cell, logged $(date '+%H:%M') BEFORE any VAL score): ${CELL[$S]}" >> temp/e381/cell.err 2>&1
done
if [ "${CELL[0]}" == "${CELL[1]}" ]; then VCELL=${CELL[0]}; say "the heads agree: verdict cell $VCELL"
else VCELL=$($PY jtoken_eval.py choose-cell --runs e381_n_s0,e381_n_s1 --split train --every 3 2>>temp/e381/cell.err | tail -1); say "the heads differ: the recipe cell (mean of the two TRAIN tables) = verdict cell $VCELL"
  for S in 0 1; do python3 ../autoresearch/tools/ledger.py note E381.n_s$S --notes "the heads differ: recipe cell (mean of both TRAIN tables, logged before any VAL score) = verdict cell $VCELL" >> temp/e381/cell.err 2>&1; done; fi
case "$VCELL" in *,*,*,on|*,*,*,off) ;; *) say "no valid cell: stop"; exit 1;; esac
flags() { IFS=, read a b c g <<< "$1"; echo "--link-thr $a --link-thr-long $b --keep-thr $c $([ "$g" == off ] && echo --no-none-gate)"; }
# (6) the VAL reads
for S in 0 1; do
  $RUN --id E381.read_n_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e381_n_s$S --seed $S $(flags $VCELL) --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off && $PY jtoken_eval.py anatomy --run e381_n_s$S --seed $S --link-thr-long 0.2 --keep-thr 0.6" >> temp/e381/read_s$S.out 2>&1
  say "read_n_s$S (verdict cell $VCELL) exit $?"
  if [ "$VCELL" != "0.3,0.2,0.6,off" ]; then
    $RUN --id E381.d115_n_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e381_n_s$S --seed $S $(flags 0.3,0.2,0.6,off) --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off" >> temp/e381/d115_s$S.out 2>&1
    say "d115_n_s$S exit $?"
  fi
  if [ "${CELL[$S]}" != "$VCELL" ]; then
    $RUN --id E381.own_n_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e381_n_s$S --seed $S $(flags ${CELL[$S]}) --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off" >> temp/e381/own_s$S.out 2>&1
    say "own_n_s$S (its own cell ${CELL[$S]}, reported) exit $?"
  fi
done
say "done"
