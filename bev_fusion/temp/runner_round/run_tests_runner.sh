#!/bin/bash
# scratch: the fast tests of runner/ with their real exit codes
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for t in tests/test_runner_class.py tests/test_suites_conform.py; do
  echo "=== $t"
  $PY $t 2>&1 | tail -12 | cut -c1-300
  echo "exit ${PIPESTATUS[0]}"
done
