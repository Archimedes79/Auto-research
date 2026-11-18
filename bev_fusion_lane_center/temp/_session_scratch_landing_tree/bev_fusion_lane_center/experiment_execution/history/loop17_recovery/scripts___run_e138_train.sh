#!/bin/bash
# E138: map-conditioned builder, training (6 u from the champion, random map situation + perception loss on purpose)
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
bash $S/_py14.sh p17_mapft.py --name e17_map_a --u 6 --lr 5e-4 > $L/e138_train.log 2>&1
echo finished > $L/e138_train.done
