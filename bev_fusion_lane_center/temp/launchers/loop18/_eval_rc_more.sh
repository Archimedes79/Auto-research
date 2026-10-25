#!/bin/bash
# after E219.j1rc and E219.rc1s1 finish: fork-pruned VAL evaluation of both (json in runs/e18/rc_eval), plus the j1 base at the same thresholds for the comparison
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
ev() { # name M K0 weight tag
  until [ -f $R/runs/e123/$1/model.pt ] && [ -f $R/runs/e123/$1/status.json ]; do sleep 10; done
  sleep 5
  bash $R/tools/env/_py14.sh p18_variant_prune.py --name $1 --M $2 --K0 $3 --weight $4 --tag $5 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_$5.log 2>&1
  cat $R/runs/e18/rc_eval_$5.log
}
ev e18_D7_j1rc 192 96 3 j1rc
bash $R/tools/env/_py14.sh p18_variant_prune.py --name e123_D7_r3_j1 --M 192 --K0 96 --tag j1p4 --thr 0.3,0.4 --json_thr 0.4 > $R/runs/e18/rc_eval_j1p4.log 2>&1
cat $R/runs/e18/rc_eval_j1p4.log
ev e18_D7_rc1s1 288 144 3 rc1s1
