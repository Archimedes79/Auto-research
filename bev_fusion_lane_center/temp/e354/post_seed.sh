#!/bin/bash
# E354 post-processing of ONE seed (card section 2, "dump + score"): the read-outs of arm r (stream, off, shuffle, trunc), arm z (off), the layer-1 control
# (the new driver on E341, state off, on the GPU), the scores unsnapped / snapped at p >= 0.1, the flip rates. Started through run.py by post_start.sh
# after the two trainings of the seed. usage: post_seed.sh <seed>. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=$1
L=temp/e354/post_s$S.log
PY="bash tools/env/_py14.sh"
say() { echo "$(date +%H:%M:%S) $*" >> $L; }
fail=0
step() { say "run: $*"; "$@" >> temp/e354/post_s${S}_out.log 2>&1; rc=$?; say "exit $rc: $1 $2 $3 $4 $5 $6"; [ $rc -ne 0 ] && fail=1; return $rc; }
say "post seed $S starts"
# dumps (GPU, batch 1, stream order; the driver waits for GPU room itself)
step $PY jrecur_eval.py dump --run e341_p_s$S --seed $S --state off          # layer-1 control -> runs/e341_p_s${S}__e354drv
step $PY jrecur_eval.py dump --run e354_z_s$S --seed $S --state off          # the control arm -> runs/e354_z_s$S
for M in stream off shuffle trunc; do
  step $PY jrecur_eval.py dump --run e354_r_s$S --seed $S --state $M         # stream -> runs/e354_r_s$S, others -> runs/e354_r_s${S}__$M
done
# scores (CPU): unsnapped and snapped at p >= 0.1
for F in e341_p_s${S}__e354drv e354_z_s$S e354_r_s$S e354_r_s${S}__off e354_r_s${S}__shuffle e354_r_s${S}__trunc; do
  [ -f runs/$F/ends_val.pkl ] || { say "no dump for $F"; fail=1; continue; }
  step $PY jends_eval.py score --run $F --seed $S --thr 0.1
  step $PY jends_eval.py score --run $F --seed $S --thr 0.1 --snap
done
# flip rates: r against z at matched delivery; the nulls of r's own checkpoint against its stream read-out
step $PY jflip.py compare --run e354_r_s$S --control e354_z_s$S --seed $S
step $PY jflip.py compare --run e354_r_s${S}__shuffle --control e354_r_s$S --seed $S
step $PY jflip.py compare --run e354_r_s${S}__off --control e354_r_s$S --seed $S
step $PY jflip.py rate --run e354_z_s$S --seed $S
step $PY jflip.py rate --run e354_r_s$S --seed $S
say "post seed $S done, fail=$fail"
exit $fail
