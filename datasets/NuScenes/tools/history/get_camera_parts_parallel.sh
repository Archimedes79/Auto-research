#!/usr/bin/env bash
# One-off: run get_camera_blob_part.sh for several parts in parallel, one log per part next to this script.
# Run from the problem folder: wsl bash temp/loop19/get_camera_parts_parallel.sh 02 03 04 05 06
here="$(cd "$(dirname "$0")" && pwd)"
for p in "$@"; do
  bash "$here/get_camera_blob_part.sh" "$p" > "$here/get_camera_part_$p.log" 2>&1 &
done
wait
grep -h "done\|MISMATCH\|ABBRUCH\|rror" "$here"/get_camera_part_*.log
df -h "$HOME" | tail -1
