#!/usr/bin/env bash
# Unpack a downloaded nuScenes archive part (.tgz) after refusing absolute or '..' paths.
#   extract_blob.sh <archive.tgz> [--delete]
# Environment:
#   NUSC_DEST     target root, REQUIRED (the raw data lives on D:, e.g. /mnt/d/Auto-research/NuScenes/...; ~/data/nuscenes_full is a folder of symlinks, never a target)
#   NUSC_MEMBERS  space-separated top-level members to unpack, e.g. "sweeps" (default: everything)
#   NUSC_WINTAR=1 unpack with Windows' tar.exe (for a target on a Windows drive such as /mnt/d: native NTFS
#                 writes of many small files are far faster than writing them through WSL)
# --delete removes the archive after a successful unpack AND a comparison with the data: verify_archive.ps1 (native Windows) checks that every member (below
# NUSC_MEMBERS, if set) exists in NUSC_DEST with the same size and the same bytes; the archive is deleted only when that shows 0 differences (README, "Archives").
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
f="$1"
dest="${NUSC_DEST:?set NUSC_DEST to the target root on D: (see data/NuScenes/README.md)}"
members="${NUSC_MEMBERS:-}"
mkdir -p "$dest"
echo "$(date +%T) path check $(basename "$f")"
if tar -tzf "$f" | grep -E '(^|/)\.\.(/|$)|^/' >/dev/null; then echo "ABORT: unsafe paths in the archive"; exit 1; fi
echo "$(date +%T) unpack to $dest ${members:+(only: $members)}"
if [ "${NUSC_WINTAR:-0}" = "1" ]; then
  # shellcheck disable=SC2086
  /mnt/c/Windows/System32/tar.exe -xzf "$(wslpath -w "$f")" -C "$(wslpath -w "$dest")" $members
else
  # shellcheck disable=SC2086
  tar -xzf "$f" -C "$dest" $members
fi
echo "$(date +%T) unpacked $(basename "$f")"
if [ "${2:-}" = "--delete" ]; then
  lst="$f.verify.txt"; only=""
  for m in $members; do only="$only;${m%/}/"; done
  printf '%s|%s|%s\n' "$(wslpath -w "$f")" "$(wslpath -w "$dest")" "${only#;}" > "$lst"
  echo "$(date +%T) compare $(basename "$f") with $dest"
  if powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$here/verify_archive.ps1")" -List "$(wslpath -w "$lst")" -Parallel 1; then
    rm -f "$f" "$lst"; echo "archive removed (compared with the data: 0 differences)"
  else
    rm -f "$lst"; echo "NOT deleted: the comparison with the data found differences or could not run"; exit 1
  fi
fi
df -h "$dest" | tail -1
