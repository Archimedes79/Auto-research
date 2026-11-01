#!/usr/bin/env bash
# E261.g (loop 22, CPU): the kill rung of E261 - rc1's threshold-only frontier at tau 0.7 / 0.2 / 0.1 / 0.05 and fork pruning 0.4 / 0.5 / 0.6 (card section 2)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
F=experiments/E261_recall-weight-6-operating-point/frontier_scratch.py
for fp in 0.4 0.5 0.6; do
  bash tools/env/_py14.sh $F e123_D7_r3_rc1:288:e123_D7_r3_rc1 0.7,0.2,0.1,0.05 fork_prune=$fp || exit 1
done
