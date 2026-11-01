#!/bin/bash
# selftests on a fresh export of HEAD (side job: 2 threads)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$(dirname "$0")/head_export/autoresearch" || exit 9
for t in "tools/selftest.py" "-m trainsuite selftest" "evalsuite/selftest.py" "evalsuite/selftest_comparison.py" "evalsuite/selftest_cost.py" "tools/archdraw.py --selftest"; do
  echo "=== $t"; timeout 900 python3 $t 2>&1 | tail -4; echo "exit ${PIPESTATUS[0]}"
done
