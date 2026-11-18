#!/bin/bash
# route attribute along the edges: tokens resampled to <= 4 m, route heads retrained (champion and control), validation evaluation
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_resample
mkdir -p $O
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_p16_f2h_train tok_p16_f2hr_train 4.0 > $O/resample.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_p16_f2h_val_clean tok_p16_f2hr_val_clean 4.0 >> $O/resample.log 2>&1
for f in oof0 oof1 oof2; do bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_e94_a_$f tok_e94_ar_$f 4.0 >> $O/resample.log 2>&1; done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_e94_a_val_clean tok_e94_ar_val_clean 4.0 >> $O/resample.log 2>&1
export NMAX=560
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_p16_f2hr_train --name p16_f2hr_route_s0 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/champ_train.log 2>&1 &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_e94_ar_oof0,tok_e94_ar_oof1,tok_e94_ar_oof2 --name e104br --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/ctrl_train.log 2>&1 &
wait
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2hr_val_clean --split val_clean --model p16_f2hr_route_s0 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/champ_eval.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_ar_val_clean --split val_clean --model e104br --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/ctrl_eval.log 2>&1
echo done > $O/done
