#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "start $(date)"
nice -n 10 bash tools/env/_py14.sh jends_eval.py dump --run e287_p1_s0 > temp/gt2/early_dump_p1_s0.log 2>&1; echo "dump done $(date)"
nice -n 10 bash tools/env/_py14.sh ../../temp/gt2/pair_diag.py e287_p1_s0 > temp/gt2/early_diag_p1_s0.log 2>&1; echo "diag done $(date)"
nice -n 10 bash tools/env/_py14.sh jpair_pr.py score --run e287_p1_s0 --control e284_h2_s0 > temp/gt2/early_pr_p1_s0.log 2>&1; echo "pr done $(date)"
echo EARLY_DONE
