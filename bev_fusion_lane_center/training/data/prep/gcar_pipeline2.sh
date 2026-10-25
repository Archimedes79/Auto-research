#!/usr/bin/env bash
# E27 Schritt 2: Teile 1-3: zuerst Validierung (alle 1.706) und jedes 2. Trainings-Sample (gerade Zeilen, alle Szenen), danach die ungeraden Trainings-Samples.
R=/mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/run_gcar.sh
L=/mnt/c/ML-Experiment/out
echo "== $(date +%H:%M:%S) Validierung + Training gerade (parallel)"
bash $R gcar_tools.py extract --split b123 --part val --workers 3 > $L/gcar_extract_val.log 2>&1 &
bash $R gcar_tools.py extract --split b123 --part train --start 0 --stride 2 --workers 3 > $L/gcar_extract_tr_even.log 2>&1 &
wait
echo "== $(date +%H:%M:%S) VAL+TRAIN-GERADE FERTIG"
touch $L/gcar_even_done.flag
bash $R gcar_tools.py extract --split b123 --part train --start 1 --stride 2 --workers 3 > $L/gcar_extract_tr_odd.log 2>&1
echo "GCAR-EXTRAKTION FERTIG $(date +%H:%M:%S)"
