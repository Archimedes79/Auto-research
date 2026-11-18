#!/usr/bin/env bash
# E25: Koordinatenkanaele (x, y, Radius) als zusaetzliche Eingabe des FCN gegen den Ausbeute-Verlust in der Ferne (E24).
# Fairer Kurzvergleich: FCN ohne und mit Koordinaten, je 6.000 Schritte (Rechenbudget), sonst wie E17. (Namen 'fcn8k'/'fcncoord8k' sind historisch.)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn
EXP=$HOME/exp_crn_b123
# Zeitgrenze: nach 04:05 Uhr nicht mehr starten (der Bericht muss vor 05:00 fertig sein)
if [ "$(date +%H%M)" -ge 0405 ] && [ "$(date +%H%M)" -lt 1200 ]; then echo "E25 uebersprungen (Zeitgrenze)"; exit 0; fi
export FEATS_NAME=feats32 GT_ROOT=$EXP/gt DATA_ON_CPU=1 NO_SEM=1 EST_NO_SEM=1 FEATS_MMAP=1
echo "##### E25a FCN ohne Koordinaten, 6.000 Schritte"
NET_COORD=0 ~/run_py.sh heat_align.py --feats $EXP/feats --graph $EXP/graph --out $O/b123_fcn8k --mode dist --cin 32 --no_sem --steps 6000 --lr 2e-3 --train_only \
  | grep -E "^step +(1|3000|6000) |Parameter"
echo "##### E25b FCN mit Koordinaten, 6.000 Schritte"
NET_COORD=1 ~/run_py.sh heat_align.py --feats $EXP/feats --graph $EXP/graph --out $O/b123_fcncoord8k --mode dist --cin 32 --no_sem --steps 6000 --lr 2e-3 --train_only \
  | grep -E "^step +(1|3000|6000) |Parameter"
echo "##### E25 FERTIG TRAINING"
