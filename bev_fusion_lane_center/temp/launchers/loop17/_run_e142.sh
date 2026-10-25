#!/bin/bash
# E142 (E209 in the report): UNSEEN failure types: the map builder (B2) and its no-map control (C2) trained on real perturbed rows of fog, radar_off, extr3, cam_front_off, cam_front3_off only; evaluated on night and blur
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
FP=cam_front3_off,fog,radar_off,extr3,cam_front_off
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_mapft.py --name e17_map_b2 --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 4 --lr 3e-4 --p_degrade 0.2 --fail_perts $FP > $L/e142_train_b2.log 2>&1 &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_mapft.py --name e17_map_c2 --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 4 --lr 3e-4 --p_degrade 0.2 --p_none 1.0 --p_wrong 0.0 --fail_perts $FP > $L/e142_train_c2.log 2>&1 &
wait
V=single,maxdecay:0.95
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_eval.py --out e142 --perts dark,blur --mapnet 1 --ckpt e17_map_b2 --map_mode exact --map_sigma 0 --suffix _B2m0 --variants $V > $L/e142_eval_b2.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_eval.py --out e142 --perts dark,blur --mapnet 1 --ckpt e17_map_b2 --map_mode none --suffix _B2none --variants $V > $L/e142_eval_b2none.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_eval.py --out e142 --perts dark,blur --mapnet 1 --ckpt e17_map_c2 --map_mode none --suffix _C2none --variants $V > $L/e142_eval_c2.log 2>&1
echo finished > $L/e142.done
