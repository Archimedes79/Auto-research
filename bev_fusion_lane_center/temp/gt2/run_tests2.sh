#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
for t in "grid/jends.py" "jends_eval.py selftest"; do echo "== $t"; bash tools/env/_py14.sh $t 2>&1 | tail -3; done
