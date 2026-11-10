#!/bin/bash
# scratch: check the two example job files with the suites' own check (starts nothing)
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
echo "=== Training.check ==="
python3 -m trainsuite Training.check ../bev_fusion/records/jobs/training/example_t23_lane_bce.json 2>&1 | cut -c1-200 | head -25
echo "exit ${PIPESTATUS[0]}"
echo "=== Runner.check ==="
python3 -m runsuite Runner.check ../bev_fusion/records/jobs/running/t20_val8.json 2>&1 | cut -c1-200 | head -25
echo "exit ${PIPESTATUS[0]}"
