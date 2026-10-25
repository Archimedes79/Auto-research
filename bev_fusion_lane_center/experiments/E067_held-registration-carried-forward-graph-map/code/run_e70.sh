#!/usr/bin/env bash
# E70: Sprung der Lokalisierung (Karten-Versatz springt in der Mitte jeder Szene): Nachfuehrung ohne / mit Neustart
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for J in 1.5 3.0 6.0 10.0; do
  ./run_gcar.sh jump_eval.py --jump $J --out e70_j$J 2>&1 | grep -E "^\[|Neustart|offset|share|Traceback|Error" | cut -c1-700
done
./run_gcar.sh jump_eval.py --jump 3.0 --jump_deg 4.0 --out e70_j3y4 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-700
echo "== $(date +%H:%M:%S) done"
