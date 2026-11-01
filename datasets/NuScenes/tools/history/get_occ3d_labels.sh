#!/usr/bin/env bash
# One-off (2026-10-26, approved by Mirko): the Occ3D-nuScenes labels (gts.tar.gz 2.74 GB + annotations.json 0.15 GB, NOT the 30.7 GB
# images) to D:\Auto-research\NuScenes\occ3d. Source: public Google Drive folder 1JObO75iTA2Ge5fa8D3BWC8R7yIG8VhrP of
# https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction (MIT + nuScenes non-commercial terms).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root=/mnt/d/Auto-research/NuScenes
mkdir -p "$root/downloads/occ3d" "$root/occ3d"
bash "$here/get_gdrive_file.sh" 1Wj2saxyvfsBuIbG0rshcm3tFPB6-tuKa "$root/downloads/occ3d/annotations.json"
bash "$here/get_gdrive_file.sh" 1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY "$root/downloads/occ3d/gts.tar.gz"
cp -p "$root/downloads/occ3d/annotations.json" "$root/occ3d/annotations.json"
NUSC_DEST="$root/occ3d" NUSC_WINTAR=1 bash "$here/extract_blob.sh" "$root/downloads/occ3d/gts.tar.gz"
echo "$(date +%T) all done"
