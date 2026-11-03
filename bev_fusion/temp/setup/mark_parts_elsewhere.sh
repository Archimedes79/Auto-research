#!/usr/bin/env bash
# One-off: mark parts 02-06 as handled elsewhere, so the running get_camera_blobs.sh only fetches part 01
# (the parts are downloaded in parallel by get_camera_blob_part.sh into ~/downloads/nuscenes_camera/pNN/).
set -euo pipefail
cd "$HOME/downloads/nuscenes_camera"
for p in 02 03 04 05 06; do
  echo "placeholder: part $p is downloaded in p$p/ by get_camera_blob_part.sh" > "v1.0-trainval${p}_blobs_camera.tgz.unpacked"
done
ls -la
