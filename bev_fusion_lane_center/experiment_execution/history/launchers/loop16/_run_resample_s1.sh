#!/bin/bash
# second seed of the route heads on the resampled tokens (champion and control), validation evaluation
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_resample
rm -f $O/done_s1
export NMAX=560
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_p16_f2hr_train --name p16_f2hr_route_s1 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 1 > $O/champ_train_s1.log 2>&1 &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_e94_ar_oof0,tok_e94_ar_oof1,tok_e94_ar_oof2 --name e104br_s1 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 1 > $O/ctrl_train_s1.log 2>&1 &
wait
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2hr_val_clean --split val_clean --model p16_f2hr_route_s1 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/champ_eval_s1.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_ar_val_clean --split val_clean --model e104br_s1 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/ctrl_eval_s1.log 2>&1
echo done > $O/done_s1
