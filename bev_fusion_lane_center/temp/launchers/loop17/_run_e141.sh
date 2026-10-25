#!/bin/bash
# E141 (E208 in the report): MATCHED clean reference = the same failure-half frames without any perturbation, for the champion, the gate, the map builder B and the gated B
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_series.py --clean --variants single,maxdecay:0.95 > $L/e141_series.log 2>&1 || exit 1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_gate.py apply --names e17_gate_k0,e17_gate_k1 --src single_cleanref --out gate_cleanref > $L/e141_gate.log 2>&1 || exit 1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_gate.py apply --names e17_gateB_k0,e17_gateB_k1 --src single_cleanref --out gateB_cleanref > $L/e141_gateB.log 2>&1 || exit 1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_eval.py --out e141_clean --prev 0 --variants single_cleanref,maxdecay:0.95_cleanref,gate_cleanref > $L/e141_eval_champ.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_eval.py --out e141_cleanB --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --suffix _m0 --variants single_cleanref,maxdecay:0.95_cleanref,gateB_cleanref > $L/e141_eval_B.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_eval.py --out e141_cleanB --mapnet 1 --ckpt e17_map_c --map_mode none --suffix _C --variants single_cleanref,maxdecay:0.95_cleanref > $L/e141_eval_C.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_route.py --out e141_route --prev 0 --series dark_single_cleanref,dark_maxdecay_0.95_cleanref,dark_gate_cleanref --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0 --tag_suffix _champ > $L/e141_route_champ.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_route.py --out e141_route --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --series dark_single_cleanref,dark_maxdecay_0.95_cleanref,dark_gateB_cleanref --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0 --tag_suffix _B > $L/e141_route_B.log 2>&1
echo finished > $L/e141.done
