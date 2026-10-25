#!/bin/bash
# E214 control: the failure-trained route heads (arm c) behind the ORIGINAL builder (the arms' builders were fine-tuned slightly with the builder loss x 0.05), M 192 / K0 96, spatial hold
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
SER=dark_maxdecay_0.95,blur_maxdecay_0.95,cam_front3_off_maxdecay_0.95
bash $R/tools/env/_pyr.sh p17_route.py --out e18_j2_route --set M=192 K0=96 --prev 0 --series $SER --routes e18_joint_c0,e18_joint_c1 --sigmas 0 --tag_suffix _origbuilder
