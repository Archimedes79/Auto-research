#!/bin/bash
cd "$(dirname "$0")/../.."
for t in test_jpast_train.py test_jmap_train.py; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh training/grid/$t > temp/_session_scratch_e376critic/$t.log 2>&1
  echo "exit $?" >> temp/_session_scratch_e376critic/$t.log
done
