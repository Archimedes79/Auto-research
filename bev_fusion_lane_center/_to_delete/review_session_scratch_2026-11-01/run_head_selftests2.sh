#!/bin/bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for t in evalsuite/selftest.py evalsuite/selftest_comparison.py evalsuite/selftest_cost.py; do
  echo "=== $t"; timeout 900 bash bev_fusion_lane_center/tools/env/_py14.sh bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/head_export/autoresearch/$t 2>&1 | tail -3; echo "exit ${PIPESTATUS[0]}"
done
echo "=== tools/selftest.py again"; (cd bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/head_export/autoresearch && timeout 900 python3 tools/selftest.py 2>&1 | tail -6); echo "exit $?"
