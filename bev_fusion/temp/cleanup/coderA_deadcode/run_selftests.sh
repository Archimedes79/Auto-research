#!/bin/bash
# scratch: run the module self-tests of the changed modules, 2 threads, one after the other
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for m in "$@"; do
  echo "=== $m"
  start=$(date +%s)
  if $PY "$m" > temp/deadcode/out_$(basename "$m" .py).txt 2>&1; then
    echo "PASS $m ($(( $(date +%s) - start )) s)"
  else
    echo "FAIL $m"
    tail -15 temp/deadcode/out_$(basename "$m" .py).txt
  fi
done
