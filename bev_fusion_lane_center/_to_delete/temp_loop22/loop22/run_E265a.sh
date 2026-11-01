#!/usr/bin/env bash
# E265.a (loop 22, CPU): spurs kept (assemble spur_len=0) on eight checkpoints, each paired against its own registry card (card section 2)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
for m in recall_s1 e24_j1rc_s1 e24_j1rc_s2 e24_j1rc_s3 recall fast_recall champion fast_recall_na; do
  bash tools/env/_py14.sh evaluate.py $m --asm spur_len=0 --tag sp0 --compare $m || exit 1
done
