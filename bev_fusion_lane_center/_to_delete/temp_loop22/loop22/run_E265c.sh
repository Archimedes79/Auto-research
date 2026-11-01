#!/usr/bin/env bash
# E265.c (loop 22, CPU): the control cards re-scored with tonight's code, each checked against its stored numbers (card section 2)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
ev() { bash tools/env/_py14.sh evaluate.py "$@" || exit 1; }
ev recall --tag l22 --expect Q=0.7404 lanes_found=0.7673 R@1_far=0.7714
ev recall_s1 --tag l22 --expect Q=0.7376 lanes_found=0.7695 R@1_far=0.7707
ev fast_recall --tag l22 --expect Q=0.7392 lanes_found=0.7364 R@1_far=0.7275
ev e24_j1rc_s1 --tag l22 --expect Q=0.7427 lanes_found=0.7433 R@1_far=0.7368
ev e24_j1rc_s2 --tag l22 --expect Q=0.7354 lanes_found=0.7360 R@1_far=0.7341
ev e24_j1rc_s3 --tag l22 --expect Q=0.7391 lanes_found=0.7393 R@1_far=0.7363
ev champion --tag l22 --expect Q=0.7350 lanes_found=0.6948 R@1_far=0.7087
ev fast_recall_na --tag l22 --expect Q=0.7388 lanes_found=0.7343 R@1_far=0.7264
