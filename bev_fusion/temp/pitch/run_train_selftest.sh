#!/bin/bash
# the train.py selftest (resume with accumulation 2, slim checkpoint, warm start, ...) of the working tree before the reviewed t11 code is committed: CPU only, nice 10, 6 threads (the box is free, no training runs)
cd "$(dirname "$0")/../.." || exit 1
start=$(date +%s)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(cd .. && pwd)" /usr/bin/time -v nice -n 10 ~/miniconda3/envs/gcar/bin/python -u training/train.py > temp/pitch/train_selftest.log 2>&1
rc=$?
echo "selftest exit $rc after $(( $(date +%s) - start )) s" >> temp/pitch/train_selftest.log
tail -n 25 temp/pitch/train_selftest.log
