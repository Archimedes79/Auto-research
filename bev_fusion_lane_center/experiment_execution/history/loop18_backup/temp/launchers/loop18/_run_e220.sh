#!/bin/bash
# E220: route heads trained with real failure frames (arm c of E214: route head only learns) in the best chain of loop 18 (map builder B + learned gate, exact map, fork pruning 0.3), failure series night / blur / three cameras off, hold + gate
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
ASM="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6 fork_prune=0.3"
python3 $R/tools/experiment/run.py --id E220.a --gpu-max-mb 9500 --retries 2 --basis E214,E207,E215 --seed 0 \
  --hyp "route heads trained on real failure frames lift the route success of the best chain under night / blur / lost cameras" \
  --pred "failure-half route success (map error 0): night >= 0.60, blur >= 0.80, three cameras off >= 0.55 for the failure-trained heads; clean half >= 0.85" \
  --rule "adopt the failure-trained heads as the default if the failure-half success is >= +0.10 over the original heads on all three series for both seeds and the clean half does not fall by more than 0.02" -- \
  bash $R/tools/env/_pyr.sh p17_route.py --out e18_route_best --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --series dark_gateB,blur_gateB,cam_front3_off_gateB \
    --routes p16_f2hr_route_s0,p16_f2hr_route_s1,e18_joint_c0,e18_joint_c1 --sigmas 0,2 --asm $ASM > $R/runs/e17/logs/e220_a.log 2>&1
