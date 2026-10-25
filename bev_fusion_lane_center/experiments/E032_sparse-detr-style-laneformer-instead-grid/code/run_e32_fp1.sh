#!/usr/bin/env bash
# (Dateiname aus der Kette uebernommen) E32-Fortsetzung: e32_main weitertrainieren (+6000 Schritte, gleiche Verluste) -> Diagnose Unteranpassung: wie viel bringt laengeres Training? Wartet auf das Ende von E35 (GPU-Platz).
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
until grep -q "E35 fertig" $O/e35.out 2>/dev/null; do sleep 20; done
if [ -f $O/e32_long/metrics.json ]; then echo "uebersprungen"; exit 0; fi
echo "== $(date +%H:%M:%S) e32_long"
/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh lane_former.py --name e32_long --init_from e32_main --base 48 --train_stride 2 --steps 6000 --bs 8 --lr 3e-4 --lr_trunk 0.4 --lam_fp 3 --amp 1 --mon 3000 --eval train_eval,val,test --workers 3 2>&1 | grep -E "Gewichte|Zwischenstand|\[train_eval\]|\[val\]|\[test\]|FERTIG|Error|Traceback"
echo "== $(date +%H:%M:%S) e32_long fertig"
