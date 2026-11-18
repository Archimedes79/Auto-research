#!/usr/bin/env bash
# E49 (breitere Grundlage: dec96) und E50 (Instanz-Normierung): beide Laeufe wie e30_base2 (4207 Zeilen = jede 2. Zeile, 3000 Schritte, Seed 21, stabilisiertes Rezept); starten, sobald die dec96-Extraktion fertig ist (E50 braucht sie nicht und startet sofort).
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
C="--steps 3000 --bs 8 --lr 6e-4 --beta2 0.98 --eps 1e-6 --workers 3 --dir 0.5 --fg gcar --lam 1 --wpos 1 --train_stride 2 --eval train_eval,val,test --seed 21"
echo "== $(date +%H:%M:%S) E50 (inorm) start"
$G dense_head.py --name e50_inorm $C --feats dec32,logit7 --inorm 1 > $O/e50_inorm.out 2>&1
echo "== $(date +%H:%M:%S) E50 done; waiting for dec96"
F=$HOME/exp_gcar_b123/feats
until [ -f $F/train_dec96_0_8414_s4.done ] && [ -f $F/train_dec96_2_8414_s4.done ] && [ -f $F/val_dec96_0_1706_s2.done ] && [ -f $F/val_dec96_1_1706_s2.done ]; do sleep 20; done
echo "== $(date +%H:%M:%S) E49 (dec96) start"
$G dense_head.py --name e49_dec96 $C --feats dec96,logit7 > $O/e49_dec96.out 2>&1
echo "== $(date +%H:%M:%S) E49 done"
