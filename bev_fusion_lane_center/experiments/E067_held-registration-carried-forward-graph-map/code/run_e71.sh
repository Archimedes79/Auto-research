#!/usr/bin/env bash
# E71: empfohlene Regel unter kombinierten Ereignissen (Sensorausfall 30-70 % der Szene + Sprung der Lokalisierung)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for JA in during before none; do
  ./run_gcar.sh combo_eval.py --jump 3.0 --jump_at $JA --workers 5 --out e71v_${JA}_j3 2>&1 | grep -E "^MEAN|Traceback|Error|Neustart"
done
./run_gcar.sh combo_eval.py --jump 6.0 --jump_at during --workers 5 --out e71v_during_j6 2>&1 | grep -E "^MEAN|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
