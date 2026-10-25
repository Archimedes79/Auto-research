#!/bin/bash
# one ladder run: _run_p16.sh <design> <name> <u> [extra args for p16_train.py ...]; marker out/loop2/e123/<name>.done (status in the card)
D=$1; N=$2; U=$3; shift 3
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e123
mkdir -p $O/$N
rm -f $O/$N.done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_train.py --design $D --name $N --u $U "$@" > $O/$N/stdout.txt 2>&1
echo $? > $O/$N.done
