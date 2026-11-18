#!/bin/bash
# control arm R (E122): trace network E94 from the shared stem at fidelity u (1 u = 600 iterations x 16 frames), then VAL subset evaluation.  usage: _run_e122.sh <seed> <u>
S=$1; U=$2
IT=$((600 * U))
N=e122_s${S}_${U}u
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
mkdir -p $O/$N
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_train.py --name $N --iters $IT --bs 16 --seed $S --stem_init e94_a --stem_lr 0.3 --workers 5 > $O/$N/stdout.txt 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_pyr.sh p16_ctrl.py --name $N --u $U --seed $S > $O/$N/eval_val250.log 2>&1
echo done > $O/$N.done
