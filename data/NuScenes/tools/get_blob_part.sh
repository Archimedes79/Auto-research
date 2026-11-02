#!/usr/bin/env bash
# Download one nuScenes trainval archive part of one sensor kind from the public bucket, check it against its
# S3 ETag, unpack it with extract_blob.sh (path check), and mark it as unpacked. Parts run in parallel
# (one call per part, each in its own download folder); a partial archive is resumed (curl -C -).
# The archive is kept until the extraction is checked: verify_archive.ps1 -Archive <archive> -Dest $NUSC_DEST must show 0 differences, then the archive is deleted
# (README, "Archives"; `extract_blob.sh <archive> --delete` does both; make_manifest.py keeps the identity). Unpack everything (no NUSC_MEMBERS) unless a part is
# really not wanted.
# Usage (from the repository root): wsl bash data/NuScenes/tools/get_blob_part.sh <part 01..10> <camera|lidar|radar>
# Environment: NUSC_DL download root, REQUIRED (a folder on D:, by convention /mnt/d/Auto-research/NuScenes_zips); NUSC_DEST (required), NUSC_MEMBERS, NUSC_WINTAR as in extract_blob.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
p="$1"; kind="$2"
dl="${NUSC_DL:?set NUSC_DL to a download folder on D:}/nuscenes_$kind/p$p"
: "${NUSC_DEST:?set NUSC_DEST to the unpack target on D: before anything is downloaded}"
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
bash "$here/extract_blob.sh" "$f" --delete                 # unpack, compare with the archive, delete it only if 0 differences (a failure stops here, the archive stays)
touch "$f.unpacked"
echo "$(date +%T) done $name"
