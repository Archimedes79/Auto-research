#!/bin/bash
# E317 scratch: the module self-tests of the INSTALLED files, CPU, 2 threads
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
for t in training/grid/jends.py training/grid/qjunction.py evaluation/jends_eval.py training/data/lane_gt_data.py; do
  echo "== $t $( [ "$t" = evaluation/jends_eval.py ] && echo selftest )"
  if [ "$t" = evaluation/jends_eval.py ]; then bash tools/env/_py14.sh $t selftest 2>&1 | tail -n 3; else bash tools/env/_py14.sh $t 2>&1 | tail -n 3; fi
done
python3 -m py_compile training/train/jends_train.py training/grid/jends.py training/data/lane_gt_data.py training/grid/qjunction.py evaluation/jends_eval.py && echo "py_compile ok"
