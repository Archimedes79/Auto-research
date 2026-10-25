#!/usr/bin/env bash
# E58: Ortswechsel (Boston <-> Singapur): gleiche Zeilenzahl, drei Koepfe (nur Boston / nur Singapur / gemischt), Auswertung je Ort auf VAL und TEST
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
V=$1
case "$V" in
  B) LOC="--train_loc 0" ;;
  S) LOC="--train_loc 1,3" ;;
  M) LOC="" ;;
esac
./run_gcar.sh dense_head.py --name e58_$V --fg gcar --feats dec32,logit7 --steps 3000 --bs 8 --lr 6e-4 --beta2 0.98 --eps 1e-6 --workers 3 --dir 0.5 --lam 1 --wpos 1 $LOC --train_rows 2200 --eval val_b,val_s,test_b,test_s --seed 31 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|ROLLBACK|Eingangs"
