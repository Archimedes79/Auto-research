#!/usr/bin/env bash
# E72: intermittente Ausfaelle - drei kurze Ausfallfenster je Szene (10-25 %, 40-55 %, 70-85 %), ohne Sprung / mit einem Sprung von 3 m mitten im zweiten Fenster
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
W=0.10-0.25,0.40-0.55,0.70-0.85
for JA in none during; do
  ./run_gcar.sh combo_eval.py --jump 3.0 --jump_at $JA --windows $W --workers 5 --out e72_${JA}_j3 2>&1 | grep -E "^MEAN|Traceback|Error|Neustart"
done
echo "== $(date +%H:%M:%S) done"
