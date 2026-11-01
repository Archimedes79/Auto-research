#!/bin/bash
# full runs (3000 steps, 1500 samples) of the timed-velocity test: label reference (same weights as boxes), boxes,
# boxes + maps; seeds 0 and 1
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for seed in 0 1; do
  for arm in "--label" "--boxes" "--boxes --maps"; do
    name=$(echo "$arm" | tr -d '-' | tr ' ' '_')
    $PY tests/test_flow_timed.py $arm --seed $seed > temp/checks/sup/full_${name}_s${seed}.log 2>&1
    echo "done $name s$seed exit $?"
  done
done
echo "chain finished"
