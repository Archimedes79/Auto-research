#!/usr/bin/env bash
# One-off (Mirko's OK 2026-10-28): nuScenes trainval parts 7-10, camera blobs (79.4 GB archives, kept) + radar blobs (0.8 GB).
# Only CAM_FRONT (samples + sweeps) is unpacked from the camera archives (the model uses the front camera; other cameras need
# a model change and disk space), all radars are unpacked. Raw data goes to D: (datasets/NuScenes/README.md rule).
cd "$(dirname "$0")/../../.." || exit 1
export NUSC_DL=/mnt/d/Auto-research/NuScenes/downloads
export NUSC_DEST=/mnt/d/Auto-research/NuScenes/nuscenes_full
export NUSC_WINTAR=1
sleep 36000 &                         # keeps the WSL VM alive
(
  NUSC_MEMBERS="" bash datasets/NuScenes/tools/get_parts_parallel.sh radar 07 08 09 10
  NUSC_MEMBERS="samples/CAM_FRONT sweeps/CAM_FRONT" bash datasets/NuScenes/tools/get_parts_parallel.sh camera 07 08 09 10
) > bev_fusion/temp/data/download_p7_10.log 2>&1
echo "download script finished $(date +%T)" >> bev_fusion/temp/data/download_p7_10.log
