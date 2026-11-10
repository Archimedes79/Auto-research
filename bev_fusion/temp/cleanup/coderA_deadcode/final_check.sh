#!/bin/bash
# scratch: final re-check of the files edited after the first gate run
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P=~/miniconda3/envs/gcar/bin/python
$P -m py_compile training/*.py training/legacy/s1_camera.py tests/test_hygiene.py tests/test_layers.py tests/test_line_bce.py tests/test_camera_features.py && echo compiled
for t in tests/test_hygiene.py tests/test_layers.py tests/test_camera_features.py training/legacy/s1_camera.py training/losses.py; do
  if $P $t > temp/deadcode/final_out.txt 2>&1; then echo "PASS $t"; else echo "FAIL $t"; tail -n 6 temp/deadcode/final_out.txt; fi
done
