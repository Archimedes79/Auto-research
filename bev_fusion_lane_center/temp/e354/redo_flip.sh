#!/bin/bash
# E354: the three flip-rate comparisons of one seed, re-run after the fix of the duplicate key in jflip.py compare (2026-11-15 23:55).
# usage: redo_flip.sh <seed>   (waits until no jflip.py process of the post job runs)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=$1
while ps -eo cmd | grep -E "[p]ython jflip.py" | grep -q .; do sleep 10; done
PY="bash tools/env/_py14.sh"
L=temp/e354/redo_flip_s$S.log
for a in "e354_r_s$S e354_z_s$S" "e354_r_s${S}__shuffle e354_r_s$S" "e354_r_s${S}__off e354_r_s$S"; do
  set -- $a
  $PY jflip.py compare --run $1 --control $2 --seed $S >> $L 2>&1
  echo "$(date +%H:%M:%S) exit $?: compare $1 against $2" >> $L
done
