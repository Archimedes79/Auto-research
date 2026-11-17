#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
(bash tools/env/_py14.sh test_jmap_train.py > $W/g_tm.txt 2>&1; echo "exit $?" >> $W/g_tm.txt) &
(bash tools/env/_py14.sh test_lane_map.py > $W/g_tl.txt 2>&1; echo "exit $?" >> $W/g_tl.txt) &
(bash tools/env/_py14.sh test_jmap.py > $W/g_tj.txt 2>&1; echo "exit $?" >> $W/g_tj.txt) &
wait
