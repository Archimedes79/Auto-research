#!/bin/bash
# seed check of the route-head comparison: a second training run (seed 1) of the route head on the traced tokens (E104b recipe) and on the refined tokens (E121c recipe)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
rm -f $O/seed1.done
mkdir -p $O/e104b_s1 $O/e121c_s1
( bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_e94_a_oof0,tok_e94_a_oof1,tok_e94_a_oof2 --name e104b_s1 --iters 40000 --bs 8 --lr 0.0007 --workers 2 --xattn --seed 1 > $O/e104b_s1/run.log 2>&1
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_a_test_clean --split test_clean --model e104b_s1 --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong > $O/e104b_s1/eval_test.log 2>&1 ) &
( bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_e120_a_oof0,tok_e120_a_oof1,tok_e120_a_oof2 --name e121c_s1 --iters 40000 --bs 8 --lr 0.0007 --workers 2 --xattn --seed 1 > $O/e121c_s1/run.log 2>&1
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e120_a_test_clean --split test_clean --model e121c_s1 --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong > $O/e121c_s1/eval_test.log 2>&1 ) &
wait
echo done > $O/seed1.done
