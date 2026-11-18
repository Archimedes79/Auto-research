#!/usr/bin/env bash
# Loop 3, E43: gemeinsamer Kopf mit Karten-Vorwissen (exakte Pose) UND Vorgaenger-Bild (Variante A) -> Gesamtsystem mit allen gewaehlten Bausteinen.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
if [ ! -f $O/e43_joint/metrics.json ]; then
  echo "== $(date +%H:%M:%S) e43_joint"
  $G dense_ext.py --name e43_joint --init_from e37_final --steps 2000 --lr 5e-4 --bs 8 --workers 3 --eval train_eval,val,test --prior map --temporal A --t_gap 1 2>&1 | grep -E "Eingangs|\[train_eval\]|\[val\]|\[test\]|FERTIG|Error|Traceback" | cut -c1-300
fi
$G sys_eval.py e43_joint 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-300
$G final_system.py --dense e43_joint --no_prop --tau_l 0.3 --out e43 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
echo "== $(date +%H:%M:%S) e43_joint fertig"
