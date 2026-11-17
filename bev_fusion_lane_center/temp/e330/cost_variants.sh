#!/bin/bash
# scratch: parameters and MACs of the arm variants (evaluation/jmodel_cost.py, CPU)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
O=temp/e330/cost_variants.txt
: > $O
run() { echo "== $*" >> $O; bash tools/env/_py14.sh jmodel_cost.py "$@" 2>&1 | grep -E "^total|junction maps head|pair head|maps: " >> $O; }
run --d 64 --evidence 1 --scales 1
run --d 64 --ctx dil+nopaint --evidence 0 --scales 1
run --d 64 --evidence 0 --scales 1
run --d 48 --evidence 1 --scales 1
run --d 40 --evidence 1 --scales 1
run --d 32 --evidence 1 --scales 1
run --d 40 --ctx dil+nopaint --evidence 0 --scales 1
echo DONE >> $O
