#!/bin/bash
# reviewer: the tests that touch the data-layer edits (lane_tokcache.build / build_env / report guard, lane_past_data.cache_dirs, jlive_train.resolve); 'selftests' = without the mutant suites
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for t in "lane_tokcache.py selftest" "test_lane_tokcache.py selftests" "test_lane_past.py selftests" "test_jlive_train.py selftests" "test_build_tokcache_root.py"; do
  s=$(date +%s)
  out=$(nice -n 10 bash tools/env/_py14.sh $t 2>&1); rc=$?
  echo "$(date +%H:%M:%S) rc=$rc $t ($(( $(date +%s) - s )) s): $(echo "$out" | tail -1 | cut -c1-150)"
done
echo "all done"
