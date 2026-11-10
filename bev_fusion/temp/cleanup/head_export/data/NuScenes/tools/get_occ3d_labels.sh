#!/usr/bin/env bash
# Recipe (first run 2026-10-26, approved by Mirko; a new download needs his OK again): the Occ3D-nuScenes labels (gts.tar.gz 2.74 GB +
# annotations.json 0.15 GB, NOT the 30.7 GB images) to D:\Auto-research\NuScenes\occ3d. Source: public Google Drive folder
# 1JObO75iTA2Ge5fa8D3BWC8R7yIG8VhrP of https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction
# (MIT + nuScenes non-commercial terms). Usage (from the repository root): wsl bash data/NuScenes/tools/get_occ3d_labels.sh
# Environment (optional): NUSC_ROOT, the raw data root on D: (default /mnt/d/Auto-research/NuScenes); NUSC_ZIPS, where the downloaded files lie until they are
# unpacked and compared (default: NuScenes_zips next to the data root; both files are deleted afterwards, README "Archives" - the first run's files were deleted on
# 2026-11-02, decision D42, the hashes of both stay in manifest.json under occ3d.files)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="${NUSC_ROOT:-/mnt/d/Auto-research/NuScenes}"
zips="${NUSC_ZIPS:-$(dirname "$root")/NuScenes_zips}"
mkdir -p "$zips/occ3d" "$root/occ3d"
bash "$here/get_gdrive_file.sh" 1Wj2saxyvfsBuIbG0rshcm3tFPB6-tuKa "$zips/occ3d/annotations.json"
bash "$here/get_gdrive_file.sh" 1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY "$zips/occ3d/gts.tar.gz"
cp -p "$zips/occ3d/annotations.json" "$root/occ3d/annotations.json"
NUSC_DEST="$root/occ3d" NUSC_WINTAR=1 bash "$here/extract_blob.sh" "$zips/occ3d/gts.tar.gz" --delete       # compares the unpack with the archive, deletes it only if 0 differences
cmp "$zips/occ3d/annotations.json" "$root/occ3d/annotations.json" && rm -f "$zips/occ3d/annotations.json"   # the other download is a plain file: identical copy, then the original may go
echo "$(date +%T) all done"
