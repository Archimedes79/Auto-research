#!/bin/bash
# E357 (T0), restart of stages 5 + 6 of temp/e357/chain.sh at 2026-11-16 01:55 (lane lead, supervision).
# Why: both trainings died with torch.AcceleratorError "CUDA error: unknown error" (p_s0 at step 1667, 01:29; p_s1 at step 2503, 01:33), the chain stopped at "trainings exit 1 / 1".
# Both runs keep their checkpoints (every 500 steps, resume=True: p_s0 continues from step 1500, p_s1 from 2500, the batches are deterministic in the step).
# Same command lines, same ids, same flags as chain.sh (gate result: --bs 8 --steps 3000); only run.py --retries 0 -> 2 (a further CUDA event resumes from the checkpoint by itself).
# The card, the rule and the data do not change. Never edit this script while it runs. Log: temp/e357/chain.log (lines "resume ...").
cd "$(dirname "$0")/../.."
source <(sed -n 8,17p temp/e357/chain.sh)           # cd, BFLC_SIG, L, PY, RUN, say, HYP, PRED, RULE, card_job (verbatim from the original chain)
exec 9>/tmp/e357_chain.lock
flock -n 9 || { say "resume: chain already runs"; exit 0; }
say "resume: the two trainings continue from their checkpoints (--bs 8 --steps 3000, run.py --retries 2)"
for S in 0 1; do
  ( SEED=$S card_job E357.p_s$S --retries 2 --gpu --ram-mb 6000 -- $PY jtok_train.py --name e357_p_s$S --seed $S --lattice 2.5 --steps 3000 --bs 8 --lr 1e-3 --workers 4 >> temp/e357/train_s$S.out 2>&1; echo $? > temp/e357/train_s$S.rc ) &
  sleep 5
done
wait
say "resume: trainings exit $(cat temp/e357/train_s0.rc) / $(cat temp/e357/train_s1.rc)"
[ "$(cat temp/e357/train_s0.rc)" = 0 ] && [ "$(cat temp/e357/train_s1.rc)" = 0 ] || exit 1
for S in 0 1; do
  SEED=$S card_job E357.read_s$S -- bash -c "$PY jtok_eval.py dump --run e357_p_s$S --seed $S && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.5 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.3 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.7 && $PY jtok_eval.py dump --run e357_p_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.5 --drop 0.17 --drop-seed 0" >> temp/e357/chain_out.log 2>&1
  say "resume: read-out seed $S exit $?"
done
say "resume: chain done"
