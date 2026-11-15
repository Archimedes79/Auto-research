#!/usr/bin/env bash
# One-off: fill the per-sample radar cache data/cache/radar for the val and train splits (resumable).
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for s in val train; do
  $PY -u training/feature_cache.py radar ~/data/nuscenes_full v1.0-trainval data/cache/radar --split $s --workers ${1:-5} 2>&1 | awk 'NR % 20 == 0 || /done|Error|Traceback/'
done
echo "radar cache fill finished $(date +%T)"
