#!/bin/bash
# scratch: run the jlive tests on the CPU, 2 threads, nice; $1 = the test's argument (one <path> | selftests | none = all)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
nice -n 10 bash tools/env/_py14.sh training/grid/test_jlive_train.py "$@" > temp/e372/e372t_out.txt 2>&1
echo "EXIT $?" >> temp/e372/e372t_out.txt
grep -v '^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] ' temp/e372/e372t_out.txt | tail -40
