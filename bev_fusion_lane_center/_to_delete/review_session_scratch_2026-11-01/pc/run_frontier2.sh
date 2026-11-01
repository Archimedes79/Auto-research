#!/bin/bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P=/mnt/c/Auto-research/bev_fusion_lane_center/tools/env/_py14.sh
D=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/pc
while pgrep -f "[r]un_frontier.sh" > /dev/null; do sleep 15; done
bash $P $D/frontier.py e123_D7_r3_rc1:288:e123_D7_r3_rc1 0.3,0.4 > $D/frontier_out2.txt 2>&1
