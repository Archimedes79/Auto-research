#!/usr/bin/env bash
# One-off (2026-10-25): camera parts 01-06 to D:\Auto-research\NuScenes (archives in downloads/, sweeps unpacked by
# Windows tar.exe into nuscenes_full/sweeps/; the keyframes in samples/ are already on disk, so only sweeps/ is unpacked).
here="$(cd "$(dirname "$0")/.." && pwd)"
export NUSC_DL=/mnt/d/Auto-research/NuScenes/downloads
export NUSC_DEST=/mnt/d/Auto-research/NuScenes/nuscenes_full
export NUSC_MEMBERS=sweeps
export NUSC_WINTAR=1
bash "$here/get_parts_parallel.sh" camera 01 02 03 04 05 06
