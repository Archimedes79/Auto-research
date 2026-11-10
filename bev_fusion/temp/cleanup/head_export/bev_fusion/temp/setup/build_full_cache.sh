#!/usr/bin/env bash
# One-off: build the feature cache of the frozen camera heads for the whole nuScenes front camera (val first, then train).
# Resumable per scene (re-run the script after an interruption). Run from anywhere; log: temp/checks/feature_cache_build.log
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for s in val train; do
  nice -n 5 "$PY" training/feature_cache.py build ~/data/nuscenes_full v1.0-trainval data/cache/cam_feats --split "$s" --device cuda --batch 16 2>&1 | grep -v -i warn
done
echo "all done"
