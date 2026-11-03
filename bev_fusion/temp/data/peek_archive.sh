#!/usr/bin/env bash
# One-off: first member names of a downloaded archive (layout check before choosing NUSC_MEMBERS)
f=/mnt/d/Auto-research/NuScenes/downloads/nuscenes_camera/p06/v1.0-trainval06_blobs_camera.tgz
ls -la "$f"
tar -tzf "$f" 2>/dev/null | head -3
tar -tzf "$f" 2>/dev/null | awk -F/ '{print $1"/"$2}' | sort -u | head -20
