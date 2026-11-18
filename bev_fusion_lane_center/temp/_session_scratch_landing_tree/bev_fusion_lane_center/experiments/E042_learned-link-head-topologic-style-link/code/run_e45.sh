#!/usr/bin/env bash
# E45: TopoLogic-artiger Verbindungskopf (Geometrie x Semantik): Daten bauen (OOF-Stuecke wie E42), trainieren, auf VAL / TEST auswerten (einzelbild-System, Verfeinerer e39_main)
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
P=$O/e39_main_oof
echo "== $(date +%H:%M:%S) E45 build"
$G link_head2.py build --name e45_link --tau_e 0.3 --workers 5 --sets $P/refined_oof0.pkl:e37_fold0:oof:train,$P/refined_oof1.pkl:e37_fold1:oof:train,$P/refined_oof2.pkl:e37_fold2:oof:train 2>&1 | tail -n 4
echo "== $(date +%H:%M:%S) E45 train"
$G link_head2.py train --name e45_link --steps 4000 --w_neg 2 --data $O/e45_link/linkdata2_refined_oof0.npz,$O/e45_link/linkdata2_refined_oof1.npz,$O/e45_link/linkdata2_refined_oof2.npz 2>&1 | tail -n 6
echo "== $(date +%H:%M:%S) E45 eval"
$G link_head2.py eval --name e45_link --model e45_link --tau_e 0.3 --workers 5 --sets $O/e39_main/refined_val.pkl:e37_final:val:val,$O/e39_main/refined_test.pkl:e37_final:test:val 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
echo "== $(date +%H:%M:%S) E45 done"
