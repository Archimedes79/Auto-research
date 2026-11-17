#!/bin/bash
# E357 (T0), continuation of chain.sh after the two trainings died with a CUDA 'unknown error' (seed 0 at step 1667, seed 1 at step 2503; chain.sh used
# _py14.sh, which does not retry; _pyr.sh does). The run folders hold ckpt.pt (steps 1500 / 2500): run.py resumes a failed id, jtok_train.py resumes a folder of the
# SAME recipe. The card text of the ids is the one of the ledger rows (same hyp / pred / rule as chain.sh). Then the read-outs as in chain.sh step 6.
# Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e357/chain.log
PYR="bash tools/env/_pyr.sh"
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
HYP="Given the ground-truth nodes of the 2.5 m lattice and no types, ONE pair head recovers at least half of the gap between the best link rule and the GT links on topo F1 and joins the junction links with an F1 of at least 0.75 (stage T0 of the token design E355)."
PRED="best rule topo F1 0.82 (seen on 31 frames before the run: 0.69), hi 0.955, p 0.92 (gap share about 0.75); junction-link F1 of p 0.80; the degraded read-out loses about 0.25 topo F1"
RULE="E357 card section 0 (by hand: oracle-node replay), section-free read-out, p >= 0.5: FAIL first (both seeds gap share on topo F1 < 0.5, or junction-link F1 < 0.50), then PASS (both seeds gap share >= 0.5 AND junction-link F1 >= 0.75), else INCONCLUSIVE; VAL 873, no TEST"
card_job() { id=$1; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout 240 --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 9>/tmp/e357_chain2.lock
flock -n 9 || { say "chain2 already runs"; exit 0; }
say "chain2 starts: resume of the two trainings"
for S in 0 1; do
  ( SEED=$S card_job E357.p_s$S --gpu --ram-mb 6000 -- $PYR jtok_train.py --name e357_p_s$S --seed $S --lattice 2.5 --steps 3000 --bs 8 --lr 1e-3 --workers 4 >> temp/e357/train_s$S.out 2>&1; echo $? > temp/e357/train2_s$S.rc ) &
  sleep 5
done
wait
say "trainings (resumed) exit $(cat temp/e357/train2_s0.rc) / $(cat temp/e357/train2_s1.rc)"
[ "$(cat temp/e357/train2_s0.rc)" = 0 ] && [ "$(cat temp/e357/train2_s1.rc)" = 0 ] || exit 1
for S in 0 1; do
  SEED=$S card_job E357.read_s$S -- bash -c "$PY jtok_eval.py dump --run e357_p_s$S --seed $S && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.5 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.3 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.7 && $PY jtok_eval.py dump --run e357_p_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.5 --drop 0.17 --drop-seed 0" >> temp/e357/chain_out.log 2>&1
  say "read-out seed $S exit $?"
done
say "chain2 done"
