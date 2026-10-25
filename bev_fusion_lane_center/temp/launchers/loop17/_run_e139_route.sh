#!/bin/bash
# E139: route success of the map-conditioned builders trained on real failure frames (arm B with the exact map, arm C without), single frame and hold, SD map error 0 and 2 m for the route head
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
SER=dark_single,dark_maxdecay_0.95,blur_single,blur_maxdecay_0.95,cam_front3_off_single,cam_front3_off_maxdecay_0.95
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_route.py --out e139_route --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --series $SER --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0,2 --tag_suffix _B > $L/e139_route_B.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_route.py --out e139_route --mapnet 1 --ckpt e17_map_c --map_mode none --series $SER --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0,2 --tag_suffix _C > $L/e139_route_C.log 2>&1
echo finished > $L/e139_route.done
