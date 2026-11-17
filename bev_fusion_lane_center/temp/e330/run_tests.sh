#!/bin/bash
# scratch: the tests of the E330 - E335 changes (CPU, 2 threads)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 BFLC_SIG=1.0
for t in "grid/jmaps.py" "grid/test_jmaps_dirx.py" "grid/test_jmaps_simplify.py" "grid/jends.py" "jmaps_eval.py selftest" "jmodel_cost.py selftest" "jrule_removal.py selftest" "jends_eval.py selftest"; do
  echo "== $t"
  bash tools/env/_py14.sh $t 2>&1 | grep -v -i "warn\|detach\|float(\|_py14.sh:" | tail -3
done
