#!/usr/bin/env bash
# One-off: prefill the GT cache for the whole available nuScenes front-camera set (val first, then train).
# Resumable (existing files are skipped). Log: temp/checks/gt_cache_build.log
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for s in val train; do
  nice -n 10 "$PY" -u training/gt_cache_build.py ~/data/nuscenes_full v1.0-trainval data/cache/gt --split "$s" --workers 4 2>&1 | grep --line-buffered -v -i warn
done
echo "all done"
