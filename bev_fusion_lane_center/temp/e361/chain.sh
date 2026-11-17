#!/bin/bash
# E361 (node-type head + per-source softmax on the oracle nodes; D111): the whole chain, every job through run.py. Card: experiments/E361_node-type-head-10000-steps/experiment.md (E359 with 10000 steps).
# waits for the flag temp/e361/GO (written after the short critic pass), then: the two trainings (two GPU tickets, `_pyr.sh` = CUDA-error retry, --workers 4),
# then per seed (in parallel, CPU, niced): dump, score (decode of record + secondary decode, both read-outs), diag (attribution), the degraded read-out.
# Deadline (lane lead, D111): the read-out must be through by 12:50 (the next experiment needs the slots about 13:00): a training not through by 12:25 is stopped at its last 500-step checkpoint by hand and reported as "stopped, not a verdict".
# Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e361/chain.log
PYR="bash tools/env/_pyr.sh"
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
HYP="E359 with 3.3 times the training steps (10000 instead of 3000): the failure of the node-type head and the per-source softmax on the oracle nodes of the 2.5 m lattice (FAIL, g1 0.267 / 0.232) was undertraining: the head reaches at least half of the gap to the GT links on topo F1 (>= 0.834) and an F1 of at least 0.75 on the junction links."
PRED="topo F1 0.80 (window 0.77 - 0.85) against E359's 0.77, g1 about 0.37 (0.25 - 0.55), G2 0.60 (0.50 - 0.76), type long_out recall 0.70 (0.62 - 0.80), oracle-source top-1 0.82 (0.77 - 0.90); FAIL likeliest (about 65 %), INCONCLUSIVE 30 %, PASS 5 %"
RULE="E361 card section 0 (by hand: oracle-node replay), section-free read-out, decode of record = argmax per source kept at type probability >= 0.5: FAIL first (both seeds g1 < 0.5, or both seeds G2 < 0.50), then PASS (both seeds g1 >= 0.5 AND G2 >= 0.75), else INCONCLUSIVE; VAL 873, no TEST"
card_job() { id=$1; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout 100 --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 9>/tmp/e361_chain.lock
flock -n 9 || { say "chain already runs"; exit 0; }
say "chain waits for the flag temp/e361/GO"
until [ -f temp/e361/GO ]; do sleep 15; done
say "GO: $(cat temp/e361/GO | head -c 120)"
[ "$(date +%H%M)" -ge 1530 ] && { say "after 15:30: no GPU start"; exit 1; }
for S in 0 1; do
  [ -e runs/e361_p3_s$S ] && { say "runs/e361_p3_s$S exists"; exit 1; }
  ( SEED=$S card_job E361.p3_s$S --gpu --ram-mb 6000 -- $PYR jtok2_train.py --name e361_p3_s$S --seed $S --lattice 2.5 --steps 10000 --bs 8 --lr 1e-3 --workers 4 >> temp/e361/train_s$S.out 2>&1; echo $? > temp/e361/train_s$S.rc ) &
  sleep 5
done
wait
say "trainings exit $(cat temp/e361/train_s0.rc) / $(cat temp/e361/train_s1.rc)"
[ "$(cat temp/e361/train_s0.rc)" = 0 ] && [ "$(cat temp/e361/train_s1.rc)" = 0 ] || exit 1
for S in 0 1; do
  ( SEED=$S card_job E361.read_s$S -- bash -c "$PY jtok2_eval.py dump --run e361_p3_s$S --seed $S && $PY jtok2_eval.py score --run e361_p3_s$S --seed $S && $PY jtok2_eval.py diag --run e361_p3_s$S --seed $S && $PY jtok2_eval.py dump --run e361_p3_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok2_eval.py score --run e361_p3_s$S --seed $S --drop 0.17 --drop-seed 0" >> temp/e361/read_s$S.out 2>&1; echo $? > temp/e361/read_s$S.rc ) &
  sleep 5
done
wait
say "read-outs exit $(cat temp/e361/read_s0.rc) / $(cat temp/e361/read_s1.rc)"
say "chain done"
