#!/usr/bin/env bash
# One-off: download the nuScenes trainval camera archives (all six cameras, keyframes + 12 Hz sweeps),
# check each against its S3 ETag, unpack with ~/extract_blob.sh (path check) into ~/data/nuscenes_full.
# Archives are kept in ~/downloads/nuscenes_camera/ until the extraction is checked.
# Run from the problem folder: wsl bash temp/loop19/get_camera_blobs.sh 01 02 03 04 05 06
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dl="$HOME/downloads/nuscenes_camera"
mkdir -p "$dl"
base="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
for p in "$@"; do
  name="v1.0-trainval${p}_blobs_camera.tgz"
  f="$dl/$name"
  if [ -e "$dl/$name.unpacked" ]; then echo "$(date +%T) $name already unpacked, skip"; continue; fi
  etag="$(curl -sfI "$base/$name" | tr -d '\r' | awk 'tolower($1)=="etag:"{print $2}' | tr -d '"')"
  echo "$(date +%T) download $name (etag $etag)"
  curl -fL --retry 10 --retry-delay 20 -C - -o "$f" "$base/$name" || curl -fL --retry 10 --retry-delay 20 -C - -o "$f" "$base/$name"
  echo "$(date +%T) verify $name"
  python3 "$here/verify_s3_etag.py" "$f" "$etag"
  echo "$(date +%T) unpack $name"
  bash "$HOME/extract_blob.sh" "$f"
  touch "$dl/$name.unpacked"
done
echo "$(date +%T) all done"
ls "$HOME/data/nuscenes_full/sweeps"
