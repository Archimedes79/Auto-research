#!/usr/bin/env bash
# Loop 3, E40 zusaetzlich: das mit EXAKTER Pose trainierte Modell (e40_clean) bei wachsender Pose-Stoerung (0,3 / 0,5 / 2 m) mit und ohne Ausrichtung sowie mit veralteter Karte (30 % / 60 %) auswerten (nur Auswertung).
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
until [ -f $O/e40_clean/metrics.json ]; do sleep 20; done
EV="--steps 0 --eval val,test --workers 3 --init_from e40_clean --prior map"
run() { name="$1"; shift; if [ -f "$O/$name/metrics.json" ]; then echo "uebersprungen: $name"; return; fi; echo "== $(date +%H:%M:%S) $name $*"; $G dense_ext.py --name "$name" "$@" 2>&1 | grep -E "Eingangs|\[train_eval\]|\[val\]|\[test\]|FERTIG|Error|Traceback" | cut -c1-300; }
for sig in 03 05 2; do
  xy=$(python3 -c "print({'03':0.3,'05':0.5,'2':2.0}['$sig'])" 2>/dev/null || echo 1.0)
  run e40_clean_s${sig}_noalign $EV --eval_prior_xy $xy --eval_prior_deg $(python3 -c "print(2*$xy)" 2>/dev/null || echo 2) --eval_align 0
  run e40_clean_s${sig}_align   $EV --eval_prior_xy $xy --eval_prior_deg $(python3 -c "print(2*$xy)" 2>/dev/null || echo 2) --eval_align 1
done
run e40_clean_stale3 $EV --eval_degrade 0.3
run e40_clean_stale6 $EV --eval_degrade 0.6
echo "== $(date +%H:%M:%S) E40 extra fertig"
