#!/bin/bash
# in-sample vs out-of-sample tokens: the same route head on training tokens (stride subset) and on validation tokens
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_insample
mkdir -p $O
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh _tok_sub.py tok_p16_f2h_train tok_p16_f2h_trsub 8 > $O/sub.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh _tok_sub.py tok_e94_a_oof0 tok_e94_a_oofsub 3 >> $O/sub.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2h_trsub --split train --model p16_f2h_route_s0 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag insample > $O/champ_train.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2h_val_clean --split val_clean --model p16_f2h_route_s0 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/champ_val.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_a_oofsub --split train --model e104b --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag oofsub > $O/ctrl_train.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_a_val_clean --split val_clean --model e104b --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/ctrl_val.log 2>&1
echo done > $O/done
