#!/usr/bin/env bash
# One-off (2026-10-25): the last step of copy_raw_to_d.sh listed the wrong folder; copy the three radar archives
# (parts 4-6) to D: and write their manifest.
set -euo pipefail
src="$HOME/downloads/nuscenes_radar_456"
d=/mnt/d/Auto-research/NuScenes
mkdir -p "$d/downloads/nuscenes_radar_456"
( cd "$src" && find . -type f -printf '%p\t%s\n' | sort ) > "$d/manifests/radar_456_archives.tsv"
cp -p "$src"/*.tgz "$d/downloads/nuscenes_radar_456/"
ls -la "$d/downloads/nuscenes_radar_456/"
