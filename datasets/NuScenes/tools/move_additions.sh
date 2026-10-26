#!/usr/bin/env bash
# One-off (2026-10-25): move experiment-specific additions out of the raw nuScenes folder into
# ~/data/nuscenes_additions/<experiment>/ and leave symlinks at the old paths, so existing configs keep working.
# Refuses if a process has one of the files open. Usage: wsl bash datasets/NuScenes/tools/move_additions.sh
set -euo pipefail
raw="$HOME/data/nuscenes_full"
dst="$HOME/data/nuscenes_additions/lcr_crn"
items=(nuscenes_infos_train.pkl nuscenes_infos_val.pkl nuscenes_infos_train_crn.pkl nuscenes_infos_val_crn.pkl radar_bev_filter radar_pv_filter)
for i in "${items[@]}"; do
  if [ -L "$raw/$i" ]; then echo "$i is already a symlink, skip"; continue; fi
  if command -v lsof >/dev/null && lsof +D "$raw/$i" >/dev/null 2>&1; then echo "ABORT: $i is in use"; exit 1; fi
done
mkdir -p "$dst"
for i in "${items[@]}"; do
  [ -L "$raw/$i" ] && continue
  mv "$raw/$i" "$dst/$i"
  ln -s "$dst/$i" "$raw/$i"
  echo "moved $i -> $dst/$i (symlink left)"
done
ls -la "$raw" | grep -- "->"
