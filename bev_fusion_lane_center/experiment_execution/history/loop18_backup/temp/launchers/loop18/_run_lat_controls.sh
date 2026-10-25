#!/bin/bash
# idle-GPU stage times of the controls and weight-6 variant (alternating j1 / j1eq / j1rc / j1rc6, two rounds), pruning 0.4; run only when no other job runs
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
P=$R/tools/env/_py14.sh
for k in a b; do
  for pair in e123_D7_r3_j1:j1 e18_D7_j1eq:j1eq e18_D7_j1rc:j1rc e18_D7_j1rc6:j1rc6; do
    name=${pair%%:*}; tag=${pair#*:}
    bash $P p18_latency.py --name $name --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.4 --tag _ctl_${tag}_$k > /dev/null 2>&1
  done
done
for f in $R/runs/e18/latency/*_ctl_*.json; do echo $(basename $f) $(grep -A2 '"assembly"' $f | grep median_ms | tr -d ' ,') $(grep -A2 '"total"' $f | grep median_ms | tr -d ' ,') $(grep points_per $f | tr -d ' ,'); done
