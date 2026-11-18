#!/usr/bin/env bash
# E65a: Merkmalsraum-Augmentierung (Sektor-Ausfall, Verstaerkung, Rauschen, Unschaerfe, Karten-Logits aus); Rezept von e37_final, danach Auswertung auf sauberen und allen gestoerten TEST-Merkmalen
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e58_rest.done ]; do sleep 30; done
echo "== $(date +%H:%M:%S) train e65_faug"
./run_gcar.sh dense_head.py --name e65_faug --fg gcar --feats dec32,logit7 --steps 6000 --bs 8 --lr 6e-4 --beta2 0.98 --eps 1e-6 --workers 3 --dir 0.5 --lam 1 --wpos 1 --faug 0.5 --seed 5 --eval train_eval,val,test 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|ROLLBACK|Eingangs|Schritt (2000|4000|6000)/"
./run_e65_eval.sh e65_faug
