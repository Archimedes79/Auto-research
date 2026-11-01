#!/usr/bin/env bash
# One-off (2026-10-25): the camera sweeps go to D: (the WSL disk image on C: has no room).
# 1. move the partial camera archives of the stopped first run from ~/downloads to D:
# 2. link ~/data/nuscenes_full/sweeps/CAM_* to the folders on D:, so every reader sees one dataroot
set -euo pipefail
droot=/mnt/d/datasets/NuScenes
mkdir -p "$droot/downloads/nuscenes_camera" "$droot/nuscenes_full/sweeps"
for p in 01 02 03 04 05 06; do
  src="$HOME/downloads/nuscenes_camera/p$p"
  dst="$droot/downloads/nuscenes_camera/p$p"
  mkdir -p "$dst"
  if [ -e "$src/v1.0-trainval${p}_blobs_camera.tgz" ]; then mv "$src/v1.0-trainval${p}_blobs_camera.tgz" "$dst/"; fi
done
raw="$HOME/data/nuscenes_full/sweeps"
for cam in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
  mkdir -p "$droot/nuscenes_full/sweeps/$cam"
  if [ -e "$raw/$cam" ] && [ ! -L "$raw/$cam" ]; then echo "ABORT: $raw/$cam exists and is not a link"; exit 1; fi
  [ -L "$raw/$cam" ] || ln -s "$droot/nuscenes_full/sweeps/$cam" "$raw/$cam"
done
ls -la "$droot"/downloads/nuscenes_camera/p0*/
ls -la "$raw" | grep CAM_
