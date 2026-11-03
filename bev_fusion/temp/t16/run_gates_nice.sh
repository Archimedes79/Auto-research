#!/bin/bash
# scratch: the gates of bev_fusion on the CPU at low priority and 2 threads while the t18.fix training runs (they must not slow the loader workers)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
nice -n 15 bash tests/run_gates.sh > temp/t16/gates3.log 2>&1
echo "gates exit $?" >> temp/t16/gates3.log
tail -5 temp/t16/gates3.log
