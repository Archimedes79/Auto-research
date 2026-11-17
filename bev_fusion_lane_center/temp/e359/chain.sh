#!/bin/bash
# E359 (node-type head + per-source softmax on the oracle nodes; D111): the whole chain, every job through run.py. Card: experiments/E359_node-type-head-per-source-softmax/experiment.md.
# waits for the flag temp/e359/GO (written after the code critic's verdict), then: the two trainings (two GPU tickets, `_pyr.sh` = CUDA-error retry, --workers 4),
# then per seed (in parallel, CPU, niced): dump, score (decode of record + secondary decode, both read-outs), diag (attribution), the degraded read-out.
# Deadline: no GPU start after 15:30 (the chain refuses to start the trainings later), everything must end by 17:15.
# Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e359/chain.log
PYR="bash tools/env/_pyr.sh"
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
HYP="With a node-type head (long_out, long_in, short_out) and a per-source softmax with a none option, one head on the oracle nodes of the 2.5 m lattice delivers the graph at least half way from the best link rule to the GT links on topo F1 (>= 0.834) and joins the junction links with an F1 of at least 0.75 (the question left open by E357)."
PRED="topo F1 0.80 (window 0.72 - 0.88), G2 link-level junction F1 0.54 (0.35 - 0.72), type head long_out F1 0.80, oracle-source top-1 0.75; FAIL likeliest (about 55 %), INCONCLUSIVE 35 %, PASS 10 %"
RULE="E359 card section 0 (by hand: oracle-node replay), section-free read-out, decode of record = argmax per source kept at type probability >= 0.5: FAIL first (both seeds g1 < 0.5, or both seeds G2 < 0.50), then PASS (both seeds g1 >= 0.5 AND G2 >= 0.75), else INCONCLUSIVE; VAL 873, no TEST"
card_job() { id=$1; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 9>/tmp/e359_chain.lock
flock -n 9 || { say "chain already runs"; exit 0; }
say "chain waits for the flag temp/e359/GO"
until [ -f temp/e359/GO ]; do sleep 15; done
say "GO: $(cat temp/e359/GO | head -c 120)"
[ "$(date +%H%M)" -ge 1530 ] && { say "after 15:30: no GPU start"; exit 1; }
for S in 0 1; do
  [ -e runs/e359_p2_s$S ] && { say "runs/e359_p2_s$S exists"; exit 1; }
  ( SEED=$S card_job E359.p2_s$S --gpu --ram-mb 6000 -- $PYR jtok2_train.py --name e359_p2_s$S --seed $S --lattice 2.5 --steps 3000 --bs 8 --lr 1e-3 --workers 4 >> temp/e359/train_s$S.out 2>&1; echo $? > temp/e359/train_s$S.rc ) &
  sleep 5
done
wait
say "trainings exit $(cat temp/e359/train_s0.rc) / $(cat temp/e359/train_s1.rc)"
[ "$(cat temp/e359/train_s0.rc)" = 0 ] && [ "$(cat temp/e359/train_s1.rc)" = 0 ] || exit 1
for S in 0 1; do
  ( SEED=$S card_job E359.read_s$S -- bash -c "$PY jtok2_eval.py dump --run e359_p2_s$S --seed $S && $PY jtok2_eval.py score --run e359_p2_s$S --seed $S && $PY jtok2_eval.py diag --run e359_p2_s$S --seed $S && $PY jtok2_eval.py dump --run e359_p2_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok2_eval.py score --run e359_p2_s$S --seed $S --drop 0.17 --drop-seed 0" >> temp/e359/read_s$S.out 2>&1; echo $? > temp/e359/read_s$S.rc ) &
  sleep 5
done
wait
say "read-outs exit $(cat temp/e359/read_s0.rc) / $(cat temp/e359/read_s1.rc)"
say "chain done"
