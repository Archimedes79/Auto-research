#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)"
PY=~/miniconda3/envs/gcar/bin/python
for t in test_host_overhead test_speed_flags test_gt_cache_targets; do
  printf "%-26s" $t
  OMP_NUM_THREADS=2 $PY tests/$t.py > /tmp/final_$t.log 2>&1; rc=$?
  echo "exit $rc | $(grep -E 'GATE|all checks pass' /tmp/final_$t.log | tail -1)"
done
cd /mnt/c/Auto-research
echo "--- files with changes vs HEAD among mine:"
git status --short bev_fusion/training/train.py bev_fusion/training/probe.py bev_fusion/training/losses.py bev_fusion/training/object_head.py bev_fusion/tests/test_speed_flags.py bev_fusion/tests/test_host_overhead.py bev_fusion/tests/run_gates.sh
echo "--- hunks of object_head.py / losses.py vs HEAD:"
git diff HEAD -U0 -- bev_fusion/training/object_head.py bev_fusion/training/losses.py | grep '^@@'
