#!/usr/bin/env bash
# E58: Wiederholung der Anpassungs-Auswertung des Singapur-Kopfes (der erste Versuch scheiterte an einem Schreibfehler), danach Abstandsklassen
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e58_train[.]sh" > /dev/null; do sleep 20; done
echo "== $(date +%H:%M:%S) train-fit e58_S (zweiter Versuch)"
./run_gcar.sh dense_head.py --name e58_S --steps 0 --fg gcar --feats dec32,logit7 --dir 0.5 --train_loc 1,3 --train_rows 2200 --eval train_own,train_eval_b,train_eval_s --workers 3 2>&1 | grep -E "^\[|Traceback|Error"
./run_gcar.sh eval_distbins.py --heads e37_final:val,test e58_B:val_b,val_s,test_b,test_s e58_S:val_b,val_s,test_b,test_s e58_M:val_b,val_s,test_b,test_s 2>&1 | tail -22 | cut -c1-400
echo "== $(date +%H:%M:%S) done"
