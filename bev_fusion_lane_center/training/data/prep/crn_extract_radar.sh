#!/usr/bin/env bash
# Radar-Archive (blobs_radar) und nur die Radar-Dateien aus den Keyframe-Archiven in ~/data/nuscenes_full entpacken.
# Aufruf: bash crn_extract_radar.sh 01 02 03     (Archive in /mnt/c/ML-Experiment/bev_fusion_lane_center/downloads)
D=/mnt/c/ML-Experiment/bev_fusion_lane_center/downloads
R=~/data/nuscenes_full
cd $R || exit 1
for N in "$@"; do
  A=$D/v1.0-trainval${N}_blobs_radar.tgz
  if [ -f "$A" ]; then
    echo "== blobs_radar $N: Inhalt (Anfang):"; tar tzf "$A" | head -3
    tar xzf "$A" --wildcards '*RADAR*'
  fi
  K=$D/v1.0-trainval${N}_keyframes.tgz
  if [ -f "$K" ]; then
    echo "== keyframes $N: nur Radar entpacken"
    tar xzf "$K" --wildcards '*RADAR*'
  fi
done
for d in samples/RADAR_* sweeps/RADAR_*; do echo "$d $(ls $d 2>/dev/null | wc -l)"; done
