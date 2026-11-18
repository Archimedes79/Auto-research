#!/bin/bash
# E121c: the cross-attention route head (E104b recipe, 40000 iterations) retrained on the tokens of the refined graph, then evaluated on the refined TEST graph
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
rm -f $O/e121c.done
mkdir -p $O/e121c
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_e120_a_oof0,tok_e120_a_oof1,tok_e120_a_oof2 --name e121c --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn > $O/e121c/run.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e120_a_test_clean --split test_clean --model e121c --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none > $O/e121c/eval_test.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e120_a_val_clean --split val_clean --model e121c --kind x --taus 0.5 --cells exact,s1,s2,s3 > $O/e121c/eval_val.log 2>&1
echo done > $O/e121c.done
