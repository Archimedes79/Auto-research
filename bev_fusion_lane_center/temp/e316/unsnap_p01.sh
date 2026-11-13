#!/bin/bash
# E316 result: the not-snapped p >= 0.1 read-out (the differentiable one, D84) of the two pair runs; day11's post scoring did not make it. CPU, nice, VAL.
cd "$(dirname "$0")/../.."
for S in 0 1; do nice bash tools/env/_py14.sh evaluation/jends_eval.py score --run e316_p_s$S --seed $S --thr 0.1 >> temp/e316/unsnap_p01_s$S.log 2>&1; done
echo done > temp/e316/unsnap_p01.done
