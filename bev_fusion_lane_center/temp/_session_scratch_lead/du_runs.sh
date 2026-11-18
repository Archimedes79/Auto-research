#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
du -sh runs temp results experiment_execution/runs _to_delete 2>/dev/null
echo "--- runs top 14 by size (MB)"
du -s runs/* 2>/dev/null | sort -rn | head -14 | awk '{printf "%8.1f  %s\n", $1/1024, $2}'
echo "--- e36x - e38x run folders: count, total MB, and how many hold BOTH ckpt.pt and model.pt"
n=0; both=0; tot=0
for d in runs/e36* runs/e37* runs/e38*; do
  [ -d "$d" ] || continue
  n=$((n+1))
  [ -f "$d/ckpt.pt" ] && [ -f "$d/model.pt" ] && both=$((both+1))
done
echo "folders $n, with ckpt.pt + model.pt: $both"
echo "--- sizes of one typical token-head run folder"
ls -la runs/e373_cb_s0 2>/dev/null | awk '{print $5, $9}' | sort -rn | head -6
echo "--- e365_cache"
du -sh runs/e365_cache/* 2>/dev/null | head -10
