#!/bin/bash
# E320 scratch: install the critic fixes atomically (cp to a temp name in the same folder, then mv), then py_compile + the self-tests of the installed files
cd "$(dirname "$0")/../.."
for pair in "jends training/grid" "jends_train training/train" "qends_eval evaluation" "jdiag evaluation"; do
  set -- $pair
  cmp -s $2/$1.py temp/e320/orig/$1.pysrc || { echo "REFUSED: $2/$1.py changed since the dev copy was made"; exit 1; }
done
for pair in "jends training/grid" "jends_train training/train" "qends_eval evaluation" "jdiag evaluation"; do
  set -- $pair
  cp temp/e320/dev/$1.pysrc $2/.$1.py.e320tmp && mv -f $2/.$1.py.e320tmp $2/$1.py && echo "installed $2/$1.py"
done
python3 -m py_compile training/train/jends_train.py training/grid/jends.py evaluation/qends_eval.py evaluation/jdiag.py && echo "py_compile ok"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh training/grid/jends.py 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/qends_eval.py selftest 2>&1 | tail -n 1
bash tools/env/_py14.sh evaluation/jdiag.py selftest 2>&1 | tail -n 1
