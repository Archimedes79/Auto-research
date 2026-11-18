#!/bin/bash
# E362 (E361 with 16000 steps; D112): the whole chain, every job through run.py. Card: experiments/E362_node-type-head-16000-steps/experiment.md
# usage: bash temp/e362/chain.sh <steps> <train_timeout_min>   (steps and timeout are set from the card v2 and the measured speed)
# two trainings (two GPU tickets, `_pyr.sh` = CUDA-error retry, --workers 4) side by side, then per seed (in parallel, CPU, niced): dump, score, diag, the degraded read-out.
# The timeout is steps x measured speed x 1.3 (the card's rule after the E361 slip), never a round number chosen for another step count.
# Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
STEPS=${1:?steps}; TO=${2:?timeout minutes}
L=temp/e362/chain.log
PYR="bash tools/env/_pyr.sh"
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
source temp/e362/texts.sh                          # HYP, PRED, RULE (the card v2 texts)
card_job() { id=$1; to=$2; shift; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout $to --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 9>/tmp/e362_chain.lock
flock -n 9 || { say "chain already runs"; exit 0; }
say "chain starts: steps $STEPS, training timeout $TO min"
for S in 0 1; do
  [ -e runs/e362_p4_s$S ] && { say "runs/e362_p4_s$S exists"; exit 1; }
  ( SEED=$S card_job E362.p4_s$S $TO --gpu --ram-mb 6000 -- $PYR jtok2_train.py --name e362_p4_s$S --seed $S --lattice 2.5 --steps $STEPS --bs 8 --lr 1e-3 --workers 4 >> temp/e362/train_s$S.out 2>&1; echo $? > temp/e362/train_s$S.rc ) &
  sleep 5
done
wait
say "trainings exit $(cat temp/e362/train_s0.rc) / $(cat temp/e362/train_s1.rc)"
[ "$(cat temp/e362/train_s0.rc)" = 0 ] && [ "$(cat temp/e362/train_s1.rc)" = 0 ] || { say "a training is not through: no read-out"; exit 1; }
for S in 0 1; do
  ( SEED=$S card_job E362.read_s$S 60 -- bash -c "$PY jtok2_eval.py dump --run e362_p4_s$S --seed $S && $PY jtok2_eval.py score --run e362_p4_s$S --seed $S && $PY jtok2_eval.py diag --run e362_p4_s$S --seed $S && $PY jtok2_eval.py dump --run e362_p4_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok2_eval.py score --run e362_p4_s$S --seed $S --drop 0.17 --drop-seed 0" >> temp/e362/read_s$S.out 2>&1; echo $? > temp/e362/read_s$S.rc ) &
  sleep 5
done
wait
say "read-outs exit $(cat temp/e362/read_s0.rc) / $(cat temp/e362/read_s1.rc)"
say "chain done"
