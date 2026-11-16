#!/bin/bash
# E320 scratch: the self-tests of the INSTALLED files, CPU, 2 threads
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
for t in training/grid/jends.py evaluation/jends_eval.py evaluation/qends_eval.py evaluation/jmaneuver.py; do
  echo "== $t"
  case $t in */jends.py) bash tools/env/_py14.sh $t 2>&1 | tail -n 2 ;; *) bash tools/env/_py14.sh $t selftest 2>&1 | tail -n 2 ;; esac
done
python3 -m py_compile training/train/jends_train.py training/grid/jends.py evaluation/jends_eval.py evaluation/qends_eval.py evaluation/jmaneuver.py && echo "py_compile ok"
bash tools/env/_py14.sh jends_train.py --help 2>&1 | grep -E -- "--attach|--geom-w|--unc" | head
