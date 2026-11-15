#!/usr/bin/env bash
# One-off (2026-10-28): when the download + unpack of parts 7-10 has finished, build the derived caches of the new scenes:
# radar cache, GT cache, feature cache of the frozen camera heads (all resumable per scene / sample; val and train splits;
# existing entries are skipped, so only the 340 new scenes cost time). Low priority: a training lane runs meanwhile.
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
until grep -q "download script finished" temp/data/download_p7_10.log 2>/dev/null; do sleep 60; done
for p in 07 08 09 10; do
  [ -f /mnt/d/Auto-research/NuScenes/downloads/nuscenes_camera/p$p/v1.0-trainval${p}_blobs_camera.tgz.unpacked ] \
    || { echo "part $p camera NOT unpacked: stop"; exit 1; }
done
echo "== data ready $(date +%T)"
for s in val train; do
  nice -n 10 $PY -u training/feature_cache.py radar ~/data/nuscenes_full v1.0-trainval data/cache/radar --split $s --workers 3 2>&1 | awk 'NR % 50 == 0 || /done|Error|Traceback/'
done
echo "== radar cache done $(date +%T)"
for s in val train; do
  nice -n 10 $PY -u training/gt_cache_build.py ~/data/nuscenes_full v1.0-trainval data/cache/gt --split $s --workers 3 2>&1 | grep --line-buffered -v -i warn | awk 'NR % 100 == 0 || /done|Error|Traceback/'
done
echo "== GT cache done $(date +%T)"
for s in val train; do
  nice -n 10 $PY training/feature_cache.py build ~/data/nuscenes_full v1.0-trainval data/cache/cam_feats --split $s --device cuda --batch 8 2>&1 | grep -v -i warn | awk 'NR % 20 == 0 || /done|Error|Traceback/'
done
echo "prep finished $(date +%T)"
