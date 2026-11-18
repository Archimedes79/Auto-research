#!/bin/bash
# scratch: the whole test suite after 22:00 (4 tests use the GPU for seconds), nice, one log
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
until [ "$(date +%H%M)" -ge 2200 ] || [ "$(date +%H%M)" -lt 0500 ]; do sleep 20; done
echo "start $(date +%H:%M:%S)" > temp/_session_scratch_prof/tests_all.log
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/maintenance/run_all_tests.sh >> temp/_session_scratch_prof/tests_all.log 2>&1
echo "end $(date +%H:%M:%S) exit $?" >> temp/_session_scratch_prof/tests_all.log
