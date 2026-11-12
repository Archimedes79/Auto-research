#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
for m in i ii iii; do
  echo "== mutation $m"; JFOLDER=mut_$m bash tools/env/_py14.sh e320/t_dev.py selftest 2>&1 | grep -v "^_py14" | grep -E "AssertionError|selftests ok|Error" | head -2
done
