#!/bin/bash
# route heads without token content, champion and control, validation evaluation (exact, 1 m, 2 m, 3 m)
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_zero_tok
mkdir -p $O
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh _zero_tok.py > $O/zero.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_p16_f2hz_train --name p16_f2hz_route_s0 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/champ_train.log 2>&1 &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_e94az_oof0,tok_e94az_oof1,tok_e94az_oof2 --name e104bz --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/ctrl_train.log 2>&1 &
wait
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2hz_val_clean --split val_clean --model p16_f2hz_route_s0 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/champ_eval.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94az_val_clean --split val_clean --model e104bz --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/ctrl_eval.log 2>&1
echo done > $O/done
