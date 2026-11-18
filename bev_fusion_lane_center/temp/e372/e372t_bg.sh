#!/bin/bash
# scratch: the full jlive test detached (survives the caller), output in temp/e372/e372t_full.txt
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
nohup setsid nice -n 10 bash tools/env/_py14.sh training/grid/test_jlive_train.py > temp/e372/e372t_full.txt 2>&1 < /dev/null &
echo started $!
