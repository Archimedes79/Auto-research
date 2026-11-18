#!/usr/bin/env bash
# E48: die zwei fehlgeschlagenen Auswertungen (sigma 0.5 und 3) mit der gelernten Registrierung wiederholen
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
run() { $G dense_ext.py --name $1 --steps 0 --init_from e40_clean --prior map --prior_xy 0 --prior_deg 0 --eval_prior_xy $2 --eval_prior_deg $3 --eval_align $4 --aligner $5 --eval val,test --workers 3 > $O/$1.out 2>&1; }
run e48_clean_s05_lalign 0.5 1 2 e48_align
run e48_clean_s3_lalign 3 6 2 e48_align
echo "== $(date +%H:%M:%S) e48 eval2 done"
