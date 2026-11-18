#!/bin/bash
# route head along the edges, point spacing 2 m and 8 m (champion graph, seed 0, validation): how fine must the attribute be sampled?
# the trainings wait for the seed-1 trainings AND for the timing measurement (an idle GPU, file timing_ok)
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_resample
P=/home/ml/exp_gcar_b123/p10
rm -f $O/done_steps
for ST in 2 8; do
  [ -f $P/tok_p16_f2hr${ST}_train.pkl ] || bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_p16_f2h_train tok_p16_f2hr${ST}_train ${ST}.0 >> $O/resample_steps.log 2>&1
  [ -f $P/tok_p16_f2hr${ST}_val_clean.pkl ] || bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_tok_resample.py tok_p16_f2h_val_clean tok_p16_f2hr${ST}_val_clean ${ST}.0 >> $O/resample_steps.log 2>&1
done
while [ ! -f $O/done_s1 ] || [ ! -f $O/timing_ok ]; do sleep 30; done
export NMAX=1100
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_p16_f2hr2_train --name p16_f2hr2_route_s0 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/champ2_train.log 2>&1 &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_train.py --tok tok_p16_f2hr8_train --name p16_f2hr8_route_s0 --iters 40000 --bs 8 --lr 0.0007 --workers 3 --xattn --seed 0 > $O/champ8_train.log 2>&1 &
wait
for ST in 2 8; do
  bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p10_route_eval2.py --tok tok_p16_f2hr${ST}_val_clean --split val_clean --model p16_f2hr${ST}_route_s0 --kind x --taus 0.5 --cells exact,s1,s2,s3 --tag val > $O/champ${ST}_eval.log 2>&1
done
echo done > $O/done_steps
