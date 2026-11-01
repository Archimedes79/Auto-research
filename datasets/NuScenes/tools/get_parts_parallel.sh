#!/usr/bin/env bash
# Run get_blob_part.sh for several parts of one sensor kind in parallel; one log per part in the download root.
# Usage (from the repository root): wsl bash datasets/NuScenes/tools/get_parts_parallel.sh <kind> <part> [<part> ...]
# The NUSC_* environment of get_blob_part.sh / extract_blob.sh is passed through.
here="$(cd "$(dirname "$0")" && pwd)"
kind="$1"; shift
logs="${NUSC_DL:?set NUSC_DL to a download folder on D:}/nuscenes_$kind"
mkdir -p "$logs"
pids=()
for p in "$@"; do
  bash "$here/get_blob_part.sh" "$p" "$kind" > "$logs/part_$p.log" 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done      # one failed part makes the whole call fail
grep -h "done\|MISMATCH\|ABORT\|rror" "$logs"/part_*.log || true
exit $rc
