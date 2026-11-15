#!/bin/bash
# E343 code step round 4: the final test run, the selftest and the real-data smoke (CPU, 2 threads, nice)
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "=== tests"; date
nice -n 10 bash tools/env/_py14.sh evaluation/tests/test_jtest_eval.py 2>&1 | tail -n 15
echo "=== selftest"; date
nice -n 10 bash tools/env/_py14.sh evaluation/jtest_eval.py selftest 2>&1 | tail -n 15
echo "=== smoke"; date
bash temp/e343/run_rehearsal_smoke_r4.sh
echo "=== all done"; date
