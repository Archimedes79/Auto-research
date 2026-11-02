#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in "evaluation/bev_fusion_area.py --selftest" "evaluation/bev_curves.py --selftest" "evaluation/bev_response.py --selftest" "evaluation/bev_gtcheck.py --selftest" "evaluation/bev_annotation_pts.py check" "evaluation/bev_selftest.py"; do
  echo "== $t"; nice -n 10 $PY $t 2>&1 | grep -E "selftest ok|passed|FAIL|Error|Traceback|annotations in|rows off" | cut -c1-160
done
echo DONE
