#!/bin/bash
# planner consumer on VALIDATION (the test frames stay untouched): champion with the old route head (vertices), champion with the route head along the edges (resampled tokens), control
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_resample
rm -f $O/done_cons
export NMAXV=560
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_chain_data.py --tok tok_p16_f2hr --route p16_f2hr_route_s0 --out p16chain_f2hr --draws 3 --workers 4 --splits val,train > $O/chain_data_f2hr.log 2>&1
for D in p16chain_f2hr p16chain_f2h chain_a; do
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p14_chain_consumer.py --data $D --out p16_consval_$D --modes route+ego --seeds 3 --eval_split val > $O/cons_$D.log 2>&1
done
echo done > $O/done_cons
