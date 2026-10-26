#!/usr/bin/env bash
# Download one nuScenes trainval archive part of one sensor kind from the public bucket, check it against its
# S3 ETag, unpack it with extract_blob.sh (path check), and mark it as unpacked. Parts run in parallel
# (one call per part, each in its own download folder); a partial archive is resumed (curl -C -).
# The archive is kept until the extraction is checked.
# Usage (from the repository root): wsl bash datasets/NuScenes/tools/get_blob_part.sh <part 01..10> <camera|lidar|radar>
# Environment (optional): NUSC_DL download root (default ~/downloads); NUSC_DEST, NUSC_MEMBERS, NUSC_WINTAR as in extract_blob.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
p="$1"; kind="$2"
dl="${NUSC_DL:-$HOME/downloads}/nuscenes_$kind/p$p"
mkdir -p "$dl"
base="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
name="v1.0-trainval${p}_blobs_${kind}.tgz"
f="$dl/$name"
if [ -e "$f.unpacked" ]; then echo "$(date +%T) $name already unpacked, skip"; exit 0; fi
etag="$(curl -sfI "$base/$name" | tr -d '\r' | awk 'tolower($1)=="etag:"{print $2}' | tr -d '"')"
echo "$(date +%T) download $name (etag $etag)"
curl -sfL --retry 10 --retry-delay 20 -C - -o "$f" "$base/$name" || curl -sfL --retry 10 --retry-delay 20 -C - -o "$f" "$base/$name"
echo "$(date +%T) verify $name"
python3 "$here/verify_s3_etag.py" "$f" "$etag"
echo "$(date +%T) unpack $name"
bash "$here/extract_blob.sh" "$f"
touch "$f.unpacked"
echo "$(date +%T) done $name"
