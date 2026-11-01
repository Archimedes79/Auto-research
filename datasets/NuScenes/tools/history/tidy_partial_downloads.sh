#!/usr/bin/env bash
# One-off (2026-10-25): after the stopped first run, put the partial part-01 archive into p01/ (the layout of
# get_blob_part.sh, so a resume finds it) and move the placeholder .unpacked markers of parts 02-06 aside.
set -euo pipefail
cd "$HOME/downloads/nuscenes_camera"
mkdir -p p01 placeholders_first_run
[ -e v1.0-trainval01_blobs_camera.tgz ] && mv v1.0-trainval01_blobs_camera.tgz p01/
for m in v1.0-trainval0*_blobs_camera.tgz.unpacked; do [ -e "$m" ] && mv "$m" placeholders_first_run/; done
ls -la . p0*/ | grep -v "^total"
