#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1
for t in "grid/jends.py" "grid/jgt.py" "grid/qjunction.py" "jpair_pr.py selftest" "jends_eval.py selftest" "jmaps_eval.py selftest"; do
  echo "== $t"; bash tools/env/_py14.sh $t 2>&1 | tail -3
done
