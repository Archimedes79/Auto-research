#!/usr/bin/env bash
# E27 Schritt 1: Speicherdateien anlegen, PCA-Basis, Mini-Extraktion, dann Teile 1-3 (Validierung + Training in zwei Haelften parallel). Protokoll: out/gcar_extract.log
R=/mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/run_gcar.sh
L=/mnt/c/ML-Experiment/out
echo "== $(date +%H:%M:%S) prepare"
bash $R gcar_tools.py prepare --split b123
bash $R gcar_tools.py prepare --split mini
echo "== $(date +%H:%M:%S) PCA"
bash $R gcar_tools.py pca --split b123 --n 300 --workers 4
echo "== $(date +%H:%M:%S) Mini"
bash $R gcar_tools.py extract --split mini --part train --workers 4
bash $R gcar_tools.py extract --split mini --part val --workers 4
echo "== $(date +%H:%M:%S) Teile 1-3: Validierung + Training (2 Haelften) parallel"
bash $R gcar_tools.py extract --split b123 --part val --workers 3 > $L/gcar_extract_val.log 2>&1 &
bash $R gcar_tools.py extract --split b123 --part train --start 0 --end 4207 --workers 3 > $L/gcar_extract_tr0.log 2>&1 &
bash $R gcar_tools.py extract --split b123 --part train --start 4207 --end 8414 --workers 3 > $L/gcar_extract_tr1.log 2>&1 &
wait
echo "GCAR-EXTRAKTION FERTIG $(date +%H:%M:%S)"
