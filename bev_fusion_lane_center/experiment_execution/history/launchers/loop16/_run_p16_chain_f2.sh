#!/bin/bash
# champion of loop 16 with the final assembly: e123_D7_r3_f1 (junction head + time recursion), inference beam 288 / 144 seeds, cov_r 0.5, vertex reuse 1.5, junction anchoring 0.5, fork relocation (peak 0.4, shift 12 m, tolerance 1.6 m)
# usage: _run_p16_chain_f2.sh <stage A|B> <tag> <tau>
export TOKSET="M=288 K0=144"
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/temp/launchers/loop16/_run_p16_chain.sh "$1" "$2" D7 e123_D7_r3_f1 "$3" cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6
