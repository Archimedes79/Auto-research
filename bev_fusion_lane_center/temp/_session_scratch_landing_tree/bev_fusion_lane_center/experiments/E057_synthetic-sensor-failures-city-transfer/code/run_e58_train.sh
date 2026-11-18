#!/usr/bin/env bash
# E58: Anpassung an das Training - die drei Koepfe (B, S, M) auf ihren eigenen Trainingszeilen und auf den Trainings-Auswertungszeilen je Ort; danach Abstandsklassen
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e58_rest.done ]; do sleep 30; done
for V in B S M; do
  case "$V" in
    B) LOC="--train_loc 0" ;;
    S) LOC="--train_loc 1,3" ;;
    M) LOC="" ;;
  esac
  echo "== $(date +%H:%M:%S) train-fit e58_$V"
  ./run_gcar.sh dense_head.py --name e58_$V --steps 0 --fg gcar --feats dec32,logit7 --dir 0.5 $LOC --train_rows 2200 --eval train_own,train_eval_b,train_eval_s --workers 3 2>&1 | grep -E "^\[|Traceback|Error"
done
./run_gcar.sh eval_distbins.py --heads e37_final:val,test e58_B:val_b,val_s,test_b,test_s e58_S:val_b,val_s,test_b,test_s e58_M:val_b,val_s,test_b,test_s 2>&1 | tail -22 | cut -c1-400
echo "== $(date +%H:%M:%S) done"
