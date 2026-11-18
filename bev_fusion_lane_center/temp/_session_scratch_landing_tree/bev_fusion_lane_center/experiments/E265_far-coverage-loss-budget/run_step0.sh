#!/bin/bash
# E265 / E266 pre-measurement of the card author (CPU, 2 threads, cached raw outputs, VAL 873); run from the problem folder:
#   wsl bash experiments/E265_far-coverage-loss-budget/run_step0.sh
# each line wrote the step0_*.txt file named on it (2026-10-31 / 11-01; machine load 8 - 14, so the [s] times are not timings)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
D=experiments/E265_far-coverage-loss-budget
bash tools/env/_py14.sh $D/far_budget.py recall > $D/step0_budget_recall.txt 2>&1                  # stage budget of the far points
bash tools/env/_py14.sh $D/union_decomp.py recall > $D/step0_union_recall.txt 2>&1                 # which part of the union loses them
bash tools/env/_py14.sh $D/far_near_miss.py recall > $D/step0_nearmiss_recall.txt 2>&1             # near misses (1 - 2 m) against absences (> 2 m)
bash tools/env/_py14.sh $D/far_near_miss.py control > $D/step0_nearmiss_control.txt 2>&1
bash tools/env/_py14.sh $D/ring_split.py recall control fast_recall > $D/step0_ring.txt 2>&1        # inner box (< 45 m) against the ring 45 - 48 m (E266)
bash tools/env/_py14.sh $D/cleanup_split.py recall fast_recall > $D/step0_cleanup.txt 2>&1         # spur against bubble removal
bash tools/env/_py14.sh $D/far_snap.py recall > $D/step0_snap_restore_recall.txt 2>&1              # range-dependent snap, restore of uncovered admitted chords
