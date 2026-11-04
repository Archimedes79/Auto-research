#!/bin/bash
# quick runs (600 steps) of the timed-velocity test, one seed: which training signal generates v?
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for arm in "--label" "--boxes" "--boxes --fwd" "--maps" "--boxes --maps"; do
  name=$(echo "$arm" | tr -d '-' | tr ' ' '_')
  $PY tests/test_flow_timed.py $arm --quick > temp/checks/sup/quick_${name}.log 2>&1
  echo "done $name exit $?"
done
echo "chain finished"
