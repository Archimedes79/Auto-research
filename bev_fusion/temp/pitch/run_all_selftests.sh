#!/bin/bash
# every selftest of the new evaluation code, CPU only
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 AR_PROJECT=../bev_fusion
PY=~/miniconda3/envs/gcar/bin/python
for t in "evaluation/bev_curves.py --selftest" "evaluation/bev_gtcheck.py --selftest" "evaluation/bev_response.py --n 50 --selftest" "../autoresearch/tools/run_report.py --selftest"; do
  echo "== $t"
  nice -n 10 $PY $t 2>&1 | grep -E "selftest ok|ego motion|rows off|Traceback|Error|assert" | cut -c1-230
done
