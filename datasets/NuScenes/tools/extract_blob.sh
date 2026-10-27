#!/usr/bin/env bash
# Unpack a downloaded nuScenes archive part (.tgz) after refusing absolute or '..' paths.
#   extract_blob.sh <archive.tgz> [--delete]
# Environment (all optional):
#   NUSC_DEST     target root (default ~/data/nuscenes_full, the WSL ext4 disk)
#   NUSC_MEMBERS  space-separated top-level members to unpack, e.g. "sweeps" (default: everything)
#   NUSC_WINTAR=1 unpack with Windows' tar.exe (for a target on a Windows drive such as /mnt/d: native NTFS
#                 writes of many small files are far faster than writing them through WSL)
# --delete removes the archive after a successful unpack.
set -euo pipefail
f="$1"
dest="${NUSC_DEST:-$HOME/data/nuscenes_full}"
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
if [ "${2:-}" = "--delete" ]; then rm -f "$f"; echo "archive removed"; fi
df -h "$dest" | tail -1
