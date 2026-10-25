#!/bin/bash
# TEST, once: the champion (final assembly) at both operating points and the control E94, graphs kept for the qualitative figure
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e123_ladder
ASM="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6"
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_final_eval.py --design D7 --name e123_D7_r3_f1 --split test_clean --set M=288 K0=144 --asm $ASM --tau 0.7 --tag finalA --keep > $O/test_champion_tau07.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_final_eval.py --design D7 --name e123_D7_r3_f1 --split test_clean --set M=288 K0=144 --asm $ASM --tau 0.3 --tag finalA03 --keep > $O/test_champion_tau03.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_ctrl.py --name e94_a --split test_clean --subset all --tag test_all --keep > $O/test_control.log 2>&1
echo done > $O/test_eval.done
