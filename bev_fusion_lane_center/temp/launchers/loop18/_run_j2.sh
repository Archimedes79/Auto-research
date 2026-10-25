#!/bin/bash
# R2.2 / E214: longer joint fine-tune (2400 steps = 2 u, bs 4) of builder + route head along the edges with REAL failure frames in half of the batches (7 types), route loss through the node tokens.
# arm a = route loss reaches the builder (builder loss x 0.05), arm c = control (tokens detached: only the route head learns), both from the same start; route-head seeds 0 / 1 (init p16_f2hr_route_s0 / s1).
# usage: _run_j2.sh <arm a|c> <seed 0|1>
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
arm=$1; s=$2
X=$R/experiment_execution/runs
until [ -f $X/E209.c2/done ]; do sleep 30; done
EXTRA=""; [ "$arm" = "c" ] && EXTRA="--no_builder_grad"
python3 $R/tools/experiment/run.py --id E214.$arm$s --gpu-max-mb 8500 --retries 3 --basis E202,E206 --seed $s \
  --hyp "joint training with failure frames: arm $arm, route seed $s" --result $R/runs/e18_joint_$arm$s/train_summary.json -- \
  bash $R/tools/env/_py14.sh p18_joint.py --name e18_joint_$arm$s --route p16_f2hr_route_s$s --iters 2400 --bs 4 --lam 0.05 --real_fail 1 --share 0.5 --seed $s $EXTRA > $R/runs/e17/logs/e214_train_$arm$s.log 2>&1
