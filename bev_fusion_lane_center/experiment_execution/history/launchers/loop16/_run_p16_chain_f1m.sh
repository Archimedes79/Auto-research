#!/bin/bash
# champion of loop 16: e123_D7_r3_f1 (junction head + time recursion), inference beam 288 / 144 seeds, assembly cov_r 0.5, vtx_r 1.5, junction anchoring 0.5, validity threshold 0.7 (chosen on the full VAL)
export TOKSET="M=288 K0=144"
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/temp/launchers/loop16/_run_p16_chain.sh "$1" f1m D7 e123_D7_r3_f1 0.7 cov_r=0.5 vtx_r=1.5 junc_thr=0.5
