#!/bin/bash
# CPU-only recomputation after the critic fixes (strict one-to-one matching, exit paint = crossings only); one thread, low priority.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "start $(date)"
(cd temp/gt2 && nice -n 10 bash ../../tools/env/_py14.sh jpainted.py > jpainted_new.log 2>&1); echo "jpainted done $(date)"
for thr in 0.5 0.3; do
  nice -n 10 bash tools/env/_py14.sh jrule_control.py score --run e284_h2_s0 e284_h2_s1 --kp-thr $thr > temp/gt2/rulectl_thr$thr.log 2>&1; echo "rule control thr $thr done $(date)"
done
nice -n 10 bash tools/env/_py14.sh jsplits.py score --run e284_h2_s0 e284_h2_s1 --kp-thr 0.5 > temp/gt2/splits_thr0.5.log 2>&1; echo "splits done $(date)"
echo CPU_ROUND_DONE
