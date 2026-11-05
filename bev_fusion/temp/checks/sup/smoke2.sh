#!/bin/bash
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=~/miniconda3/envs/gcar/bin/python
for arm in "--boxes" "--boxes --corr" "--label --corr" "--maps --corr"; do
  $PY tests/test_flow_timed.py $arm --quick --iters 2 2>&1 | grep -E "Traceback|Error|^device|parameters|claims hold|  File" | head -8
done
