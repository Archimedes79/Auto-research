#!/usr/bin/env bash
# One-off: unpack the radar archives of parts 7-10 (their first unpack failed: the PATH tar.exe was not Windows' bsdtar)
cd "$(dirname "$0")/../../.." || exit 1
export NUSC_DEST=/mnt/d/Auto-research/NuScenes/nuscenes_full NUSC_WINTAR=1
for p in 07 08 09 10; do
  f=/mnt/d/Auto-research/NuScenes/downloads/nuscenes_radar/p$p/v1.0-trainval${p}_blobs_radar.tgz
  bash datasets/NuScenes/tools/extract_blob.sh "$f" && touch "$f.unpacked"
done
echo "radar unpack finished $(date +%T)"
