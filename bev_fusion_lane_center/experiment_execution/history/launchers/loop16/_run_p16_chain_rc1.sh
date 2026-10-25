#!/bin/bash
# validation-only exploration: navigation chain stage A on the recall-weighted fine-tune rc1 (same final assembly, tau 0.7)
export TOKSET="M=288 K0=144"
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/temp/launchers/loop16/_run_p16_chain.sh A rc1v D7 e123_D7_r3_rc1 0.7 cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6
