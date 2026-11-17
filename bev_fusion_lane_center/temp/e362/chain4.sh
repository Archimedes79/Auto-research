#!/bin/bash
# E362 supervisor (written 2026-11-16 18:59 after repeated simultaneous CUDA 'unknown error' crashes of both trainings; _pyr.sh retries only 4 times per run.py call):
# per seed: whenever no run.py job E362.p4_s<S> is alive and the training is not done, start it again through run.py (it resumes from the 500-step checkpoint, same card text);
# at most 15 relaunches per seed; when both are done and chain2 has not run the read-outs, run them as chain.sh does. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e362/chain.log
PYR="bash tools/env/_pyr.sh"
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [sup] $*" >> $L; }
source temp/e362/texts.sh
card_job() { id=$1; to=$2; shift; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout $to --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 8>/tmp/e362_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts"
sup() {
  S=$1; n=0
  while ! grep -q "^[0-9:]* done e362_p4_s$S" runs/e362_p4_s$S/log.txt; do
    if pgrep -f "run.py --id E362.p4_s$S " > /dev/null; then sleep 20; continue; fi
    sleep 15
    pgrep -f "run.py --id E362.p4_s$S " > /dev/null && continue
    grep -q "^[0-9:]* done e362_p4_s$S" runs/e362_p4_s$S/log.txt && break
    n=$((n+1)); [ $n -gt 15 ] && { say "seed $S: 15 relaunches, giving up"; return 1; }
    say "seed $S: relaunch $n (no live run.py job, training not done)"
    SEED=$S card_job E362.p4_s$S 200 --gpu --ram-mb 6000 -- $PYR jtok2_train.py --name e362_p4_s$S --seed $S --lattice 2.5 --steps 16000 --bs 8 --lr 1e-3 --workers 4 >> temp/e362/train_s$S.out 2>&1
    echo $? > temp/e362/train_s$S.rc
  done
  say "seed $S: training done"
}
sup 0 &
sleep 7
sup 1 &
wait
while pgrep -f "chain2.sh" > /dev/null; do sleep 15; done
if tail -n 5 temp/e362/chain.log | grep -q "read-outs exit\|chain done"; then say "chain2 ran the read-outs: nothing to do"; exit 0; fi
say "both trainings done: read-outs by the supervisor"
for S in 0 1; do
  ( SEED=$S card_job E362.read_s$S 60 -- bash -c "$PY jtok2_eval.py dump --run e362_p4_s$S --seed $S && $PY jtok2_eval.py score --run e362_p4_s$S --seed $S && $PY jtok2_eval.py diag --run e362_p4_s$S --seed $S && $PY jtok2_eval.py dump --run e362_p4_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok2_eval.py score --run e362_p4_s$S --seed $S --drop 0.17 --drop-seed 0" >> temp/e362/read_s$S.out 2>&1; echo $? > temp/e362/read_s$S.rc ) &
  sleep 5
done
wait
say "read-outs exit $(cat temp/e362/read_s0.rc) / $(cat temp/e362/read_s1.rc)"
say "chain done"
