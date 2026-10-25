#!/bin/bash
# E135b: map assignment (route head along the edges) on the learned-gate series (recursion off, as the gate was trained)
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
bash $S/_py14.sh p17_route.py --out e135_route --prev 0 --series dark_gate,blur_gate,cam_front3_off_gate,dark_maxdecay_0.95,blur_maxdecay_0.95,cam_front3_off_maxdecay_0.95 --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0,2 > $L/e135b.log 2>&1
echo finished > $L/e135b.done
