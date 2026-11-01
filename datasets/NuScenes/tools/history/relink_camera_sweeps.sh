#!/usr/bin/env bash
# One-off (2026-10-25): the raw-data root on D: moved to D:\Auto-research\NuScenes; point the CAM_* sweep links there.
set -euo pipefail
droot=/mnt/d/Auto-research/NuScenes
raw="$HOME/data/nuscenes_full/sweeps"
for cam in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
  [ -L "$raw/$cam" ] || { echo "ABORT: $raw/$cam is not a link"; exit 1; }
  ln -sfn "$droot/nuscenes_full/sweeps/$cam" "$raw/$cam"
  [ -d "$raw/$cam/" ] || { echo "ABORT: link $cam does not resolve"; exit 1; }
done
ls -la "$raw" | grep CAM_
