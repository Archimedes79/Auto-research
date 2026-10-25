#!/bin/bash
# E232: longer joint fine-tune (4800 steps = 4 u, bs 4) of builder + route head along the edges with real failure frames in half of the batches, the builder loss at FULL weight (lam 1.0: graph quality is kept), route loss through the node tokens.
# arm a = the route loss reaches the builder, arm c = control (tokens detached: only the route head learns; the builder still learns with its own loss, same budget); route seed 0 only (time).
# usage: _run_j3.sh <arm a|c> <seed>   (starts when E231.a and E219.j1rc6 are finished: at most 2 GPU jobs)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
arm=$1; s=$2
X=$R/experiment_execution/runs
until [ -f $X/E231.a/done ] && [ -f $R/runs/e18/rc_eval/j1rc6.json ]; do sleep 30; done
sleep $((10 + 20 * $( [ "$arm" = "c" ] && echo 1 || echo 0 )))
EXTRA=""; [ "$arm" = "c" ] && EXTRA="--no_builder_grad"
python3 $R/tools/experiment/run.py --id E232.$arm$s --gpu-max-mb 8500 --retries 3 --basis E214,E202,E206 --seed $s \
  --hyp "a longer joint fine-tune with the builder loss at full weight lets the route loss through the builder beat the detached control (arm $arm, route seed $s)" \
  --pred "a - c failure-half route success < +0.03 (third negative joint result, as E202 and E214)" \
  --rule "a - c >= +0.03 in at least two of three failure types AND graph S failure half not lower by more than 0.005: joint refinement is worthwhile (replicate with seed 1); otherwise stop the joint E2E line" \
  --result $R/runs/e18_joint2_$arm$s/train_summary.json -- \
  bash $R/tools/env/_py14.sh p18_joint.py --name e18_joint2_$arm$s --route p16_f2hr_route_s$s --iters 4800 --bs 4 --lam 1.0 --real_fail 1 --share 0.5 --seed $s $EXTRA > $R/runs/e17/logs/e232_train_$arm$s.log 2>&1
