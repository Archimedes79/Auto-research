#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in "evaluation/metrics.py" "evaluation/bev_selftest.py --golden" "evaluation/bev_response.py --selftest" "evaluation/bev_gtcheck.py --selftest"; do
  echo "== $t"; nice -n 10 $PY $t 2>&1 | grep -E "selftest ok|passed|FAIL|Error|Traceback|golden|rows off|identical|differ" | cut -c1-200
done
echo DONE
