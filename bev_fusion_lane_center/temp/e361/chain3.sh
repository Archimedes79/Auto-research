#!/bin/bash
# E361 continuation (written 2026-11-16 11:25 on the lane lead's warning): chain.sh started the two trainings through run.py with `--timeout 100` (my edit from 120 when I planned 15000 steps), so run.py cuts both at about 12:06.
# This chain waits until no run.py job E361.p3_s* is left, then RESUMES every seed whose training is not done (run.py resumes a failed id from the 500-step checkpoint, `_pyr.sh`, the card text of the ledger row),
# with a timeout that ends at 12:30 (the lane lead's cut-off for training), then runs the read-outs as chain.sh would have (two seeds in parallel, CPU).
# A seed that is not done at 12:30 is reported as 'stopped at step N, not a verdict' (by hand). Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e361/chain.log
PYR="bash tools/env/_pyr.sh"
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [chain3] $*" >> $L; }
source <(sed -n 14,16p temp/e361/chain.sh)        # HYP, PRED, RULE (the same text as the ledger rows: a repeat refuses another card text)
card_job() { id=$1; to=$2; shift; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout $to --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 9>/tmp/e361_chain3.lock
flock -n 9 || { say "chain3 already runs"; exit 0; }
say "chain3 waits for the end of the run.py jobs E361.p3_s*"
while pgrep -f "run.py --id E361.p3_s" > /dev/null; do sleep 10; done
say "run.py jobs gone: $(for S in 0 1; do echo -n "s$S: $(grep -c "done e361_p3_s$S" runs/e361_p3_s$S/log.txt) done-lines, last: $(tail -n 1 runs/e361_p3_s$S/log.txt | cut -c1-60) | "; done)"
now=$(date +%s); end=$(date -d "12:30" +%s); to=$(( (end - now) / 60 ))
[ $to -lt 5 ] && { say "after 12:25: no resume (stopped, not a verdict)"; exit 1; }
for S in 0 1; do
  if grep -q "done e361_p3_s$S" runs/e361_p3_s$S/log.txt; then say "seed $S finished before the cut: no resume"; echo 0 > temp/e361/train3_s$S.rc; continue; fi
  ( SEED=$S card_job E361.p3_s$S $to --gpu --ram-mb 6000 -- $PYR jtok2_train.py --name e361_p3_s$S --seed $S --lattice 2.5 --steps 10000 --bs 8 --lr 1e-3 --workers 4 >> temp/e361/train_s$S.out 2>&1; echo $? > temp/e361/train3_s$S.rc ) &
  sleep 5
done
wait
say "trainings (resumed) exit $(cat temp/e361/train3_s0.rc) / $(cat temp/e361/train3_s1.rc)"
[ "$(cat temp/e361/train3_s0.rc)" = 0 ] && [ "$(cat temp/e361/train3_s1.rc)" = 0 ] || { say "a training is not through: stopped, not a verdict"; exit 1; }
for S in 0 1; do
  ( SEED=$S card_job E361.read_s$S 60 -- bash -c "$PY jtok2_eval.py dump --run e361_p3_s$S --seed $S && $PY jtok2_eval.py score --run e361_p3_s$S --seed $S && $PY jtok2_eval.py diag --run e361_p3_s$S --seed $S && $PY jtok2_eval.py dump --run e361_p3_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok2_eval.py score --run e361_p3_s$S --seed $S --drop 0.17 --drop-seed 0" >> temp/e361/read_s$S.out 2>&1; echo $? > temp/e361/read_s$S.rc ) &
  sleep 5
done
wait
say "read-outs exit $(cat temp/e361/read_s0.rc) / $(cat temp/e361/read_s1.rc)"
say "chain3 done"
