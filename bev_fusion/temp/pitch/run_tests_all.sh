#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in "evaluation/bev_fusion_area.py --selftest" "evaluation/bev_curves.py --selftest" "evaluation/bev_gtcheck.py --selftest" "evaluation/bev_selftest.py" "evaluation/metrics.py"; do
  echo "== $t"; nice -n 10 $PY $t 2>&1 | tail -3
done
echo "== annotation check"; nice -n 10 $PY evaluation/bev_annotation_pts.py check 2>&1 | tail -2
