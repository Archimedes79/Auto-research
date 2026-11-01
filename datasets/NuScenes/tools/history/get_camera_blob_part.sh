#!/usr/bin/env bash
# One-off: like get_camera_blobs.sh, but one part in its own download folder, so several parts run in parallel.
# Run from the problem folder: wsl bash temp/loop19/get_camera_blob_part.sh 02
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
p="$1"
dl="$HOME/downloads/nuscenes_camera/p$p"
mkdir -p "$dl"
base="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
name="v1.0-trainval${p}_blobs_camera.tgz"
f="$dl/$name"
if [ -e "$f.unpacked" ]; then echo "$(date +%T) $name already unpacked, skip"; exit 0; fi
etag="$(curl -sfI "$base/$name" | tr -d '\r' | awk 'tolower($1)=="etag:"{print $2}' | tr -d '"')"
echo "$(date +%T) download $name (etag $etag)"
curl -sfL --retry 10 --retry-delay 20 -C - -o "$f" "$base/$name" || curl -sfL --retry 10 --retry-delay 20 -C - -o "$f" "$base/$name"
echo "$(date +%T) verify $name"
python3 "$here/verify_s3_etag.py" "$f" "$etag"
echo "$(date +%T) unpack $name"
bash "$HOME/extract_blob.sh" "$f"
touch "$f.unpacked"
echo "$(date +%T) done $name"
