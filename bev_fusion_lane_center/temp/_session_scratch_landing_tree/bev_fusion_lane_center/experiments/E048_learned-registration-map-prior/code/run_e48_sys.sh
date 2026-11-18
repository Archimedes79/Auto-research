#!/usr/bin/env bash
# E48: Systemwirkung (Vorschlaege + Verfeinerer e39_main) des Vorwissens-Kopfes unter 1 m / 2 m Pose-Stoerung mit Korrelationssuche (E40) bzw. Korrelation + gelernter Feinstufe (E48); danach Graphen mit Knoten-Einbau nicht noetig
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
for r in e40_clean_s1_align e48_clean_s1_corrfine e48_clean_s2_corr e48_clean_s2_corrfine; do
  until [ -f $O/$r/metrics.json ]; do sleep 20; done
  if [ ! -f $O/sys_$r/metrics.json ]; then $G sys_eval.py $r 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-300; fi
done
echo "== $(date +%H:%M:%S) e48 sys done"
