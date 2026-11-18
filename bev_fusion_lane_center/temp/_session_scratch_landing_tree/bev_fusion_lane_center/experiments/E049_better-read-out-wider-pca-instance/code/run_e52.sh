#!/usr/bin/env bash
# E52: zwei eingefrorene Grundlagen kombiniert (GaussianCaR dec32 + logit7 + CRN-PCA32): gleiches Rezept wie e30_base2 (4207 Zeilen, 3000 Schritte, Seed 21)
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
$G dense_head.py --name e52_both --fg both --feats dec32,logit7 --steps 3000 --bs 8 --lr 6e-4 --beta2 0.98 --eps 1e-6 --workers 3 --dir 0.5 --lam 1 --wpos 1 --train_stride 2 --eval train_eval,val,test --seed 21 > $O/e52_both.out 2>&1
echo "== $(date +%H:%M:%S) e52 done"
