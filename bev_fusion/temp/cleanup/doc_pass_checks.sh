#!/bin/bash
cd /mnt/c/Auto-research || exit 1
echo "== archdraw render + check (bev_fusion training)"
python3 autoresearch/tools/archdraw.py render bev_fusion/training/architecture.json 2>&1 | tail -n 2 | cut -c1-200
python3 autoresearch/tools/archdraw.py check bev_fusion/training/architecture.json 2>&1 | tail -n 3 | cut -c1-200
git diff --stat -- bev_fusion/training/architecture.json bev_fusion/training/architecture.md bev_fusion/training/architecture.svg | cat
echo "== structure check bev_fusion"
(cd bev_fusion && python3 ../autoresearch/tools/check_structure.py 2>&1 | tail -n 4 | cut -c1-200)
echo "== structure check lane"
(cd bev_fusion_lane_center && python3 ../autoresearch/tools/check_structure.py 2>&1 | tail -n 4 | cut -c1-200)
echo "== tests"
cd bev_fusion
for t in tests/test_splits.py tests/test_gt_prepare.py; do
  printf '%s: ' $t; OMP_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python $t 2>&1 | tail -n 1 | cut -c1-160
done
