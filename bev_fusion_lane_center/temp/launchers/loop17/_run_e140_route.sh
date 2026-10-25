#!/bin/bash
# route success of the best chain: gated memory behind the map-conditioned builder (arm B, exact map; route head with the same map at error 0 and 2 m)
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_route.py --out e140_route --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --series dark_gateB,blur_gateB,cam_front3_off_gateB --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0,2 --tag_suffix _B > $L/e140_route.log 2>&1
echo finished > $L/e140_route.done
