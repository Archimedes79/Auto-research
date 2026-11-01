#!/bin/bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P=/mnt/c/Auto-research/bev_fusion_lane_center/tools/env/_py14.sh
D=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/pc
F=$D/frontier.py
{
bash $P $F e123_D7_r3_rc1:288:e123_D7_r3_rc1,e18_D7_rc2:288:e18_D7_rc2 0.5,0.6,0.8,0.9
bash $P $F e18_D7_rc2:288:e18_D7_rc2 0.7,0.8 fork_prune=0.5
bash $P $F e18_D7_rc2:288:e18_D7_rc2 0.7 vtx_r=1.7
bash $P $F control 0
bash $P $F e18_D7_f1eq:288:e18_D7_rc2,e18_D7_rc1s1:288:e18_D7_rc1s1,e18_D7_rc4:288:e18_D7_rc4 0.7
} > $D/frontier_out.txt 2>&1
