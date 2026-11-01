#!/usr/bin/env bash
# One-off (2026-10-25): copy the RAW nuScenes data from the WSL disk (on C:) to D:\Auto-research\NuScenes.
# Raw only: the archive contents (metadata, maps, samples, radar sweeps, licence) of nuscenes_full and the mini
# copy, plus the downloaded radar archives. Symlinks (the lane project's additions, the camera sweep links) are not
# copied. Streams WSL tar into Windows tar.exe (fast native writes on D:) and writes a manifest (path, size) of
# what was sent, for the check in verify_raw_copy.sh. The WSL originals stay until the copy is verified.
set -euo pipefail
d=/mnt/d/Auto-research/NuScenes
mkdir -p "$d/nuscenes_full" "$d/nuscenes_mini" "$d/downloads/nuscenes_radar_456" "$d/manifests"
copy() {  # copy <source dir> <target dir on D:> <name> <members...>
  local src="$1" dst="$2" name="$3"; shift 3
  ( cd "$src" && find "$@" -type f -printf '%p\t%s\n' | sort ) > "$d/manifests/$name.tsv"
  echo "$(date +%T) $name: $(wc -l < "$d/manifests/$name.tsv") files"
  tar -C "$src" -cf - "$@" | tar.exe -xf - -C "$(wslpath -w "$dst")"
  echo "$(date +%T) $name copied"
}
cd "$HOME/data/nuscenes_full"
radar_sweeps=$(ls -d sweeps/RADAR_*)
copy "$HOME/data/nuscenes_full" "$d/nuscenes_full" nuscenes_full LICENSE .v1.0-*.txt maps samples v1.0-trainval $radar_sweeps
cd "$HOME/data/nuscenes"
copy "$HOME/data/nuscenes" "$d/nuscenes_mini" nuscenes_mini .v1.0-mini.txt LICENSE maps samples sweeps v1.0-mini
copy "$HOME/downloads/nuscenes_radar_456" "$d/downloads/nuscenes_radar_456" radar_456_archives $(ls -A)
echo "$(date +%T) all copied"
