#!/bin/bash
# E135: map assignment (route head along the edges, 2 seeds) of the champion chain on the failure series, SD map error 0 and 2 m, single frame and spatial hold
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
bash $S/_py14.sh p17_route.py --out e135_route --series dark_single,dark_maxdecay_0.95,blur_single,blur_maxdecay_0.95,cam_front3_off_single,cam_front3_off_maxdecay_0.95 --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0,2 > $L/e135.log 2>&1
echo finished > $L/e135.done
