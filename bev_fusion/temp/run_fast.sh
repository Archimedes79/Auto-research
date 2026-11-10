#!/bin/bash
# scratch: the fast tests of my area with real exit codes
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for t in tests/test_runner_class.py tests/test_suites_conform.py tests/test_evaluation_api.py; do
  echo "=== $t"
  $PY $t 2>&1 | grep -v "^WARNING" | tail -5 | cut -c1-230
  echo "exit ${PIPESTATUS[0]}"
done
echo "=== evalsuite toy selftest with this Runner"
(cd ../autoresearch && AR_PROJECT=../bev_fusion $PY -m evalsuite.selftest_api 2>&1 | tail -3 | cut -c1-230; echo "exit ${PIPESTATUS[0]}")
