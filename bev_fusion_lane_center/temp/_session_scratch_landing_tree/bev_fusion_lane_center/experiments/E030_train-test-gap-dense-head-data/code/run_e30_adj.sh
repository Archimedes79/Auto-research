#!/usr/bin/env bash
# E30 Anpassungen gegen die Train-Val-Luecke: Referenz (gleiches Rezept) / Gewichtsabfall + Dropout / Spiegel-Augmentierung; 4207 Zeilen (jede 2. Zeile), 3000 Schritte, stabilisiertes Rezept
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
C="--steps 3000 --bs 8 --lr 6e-4 --beta2 0.98 --eps 1e-6 --workers 2 --dir 0.5 --fg gcar --feats dec32,logit7 --lam 1 --wpos 1 --train_stride 2 --eval train_eval,val,test --seed 21"
echo "== $(date +%H:%M:%S) e30 adjustments start"
$G dense_head.py --name e30_base2 $C > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e30_base2.out 2>&1 &
$G dense_head.py --name e30_reg $C --wd 1e-2 --drop 0.1 > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e30_reg.out 2>&1 &
$G dense_head.py --name e30_flip $C --flip 1 > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e30_flip.out 2>&1 &
wait
echo "== $(date +%H:%M:%S) e30 adjustments fertig"
