#!/bin/bash
# scratch: the tests that run_gates.sh runs only with --all (and the whole-chain self-test), one after the other, 2 threads
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for t in tests/test_projection_general.py training/model.py tests/test_radar_slot.py tests/test_line_bce.py tests/test_end_to_end.py; do
  start=$(date +%s)
  if $PY $t > temp/deadcode/rest_$(basename $t .py).txt 2>&1; then r=PASS; else r=FAIL; fi
  echo "$r $t ($(( $(date +%s) - start )) s): $(grep -v -i warn temp/deadcode/rest_$(basename $t .py).txt | tail -n 1 | cut -c1-170)"
done
echo REST DONE
