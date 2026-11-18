#!/usr/bin/env bash
# E18: moderner dichter Decoder (CNT-Net: ConvNeXt-V2-Encoder + Transformer-Flaschenhals, 4,8 Mio.) auf den CRN-Merkmalen, Teile 1-3.
# Gleiche Daten wie FCN (E17a) und U-Netz (E17b); nur 6.000 Schritte (Rechenbudget), AMP, Merkmale per Memory-Map.
# Die Auswertung (Schaetzer, Ausrichtung) kommt aus den Nachbearbeitungs-Skripten (exp_est.py, exp_assoc.py) auf den Abstandsfeldern.
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn
EXP=$HOME/exp_crn_b123
export FEATS_NAME=feats32 GT_ROOT=$EXP/gt DATA_ON_CPU=1 NO_SEM=1 EST_NO_SEM=1 FEATS_MMAP=1
echo "##### E18 CNT-Net base 64, 6.000 Schritte (AMP, Rechenbudget)"
~/run_py.sh heat_align.py --feats $EXP/feats --graph $EXP/graph --out $O/b123_cnt --mode dist --cin 32 --no_sem --arch cnt --base 64 \
  --steps 6000 --lr 1e-3 --amp --train_only | grep -E "^step +(1|1500|3000|4500|6000) |Parameter"
echo "##### E18 ENDE"
