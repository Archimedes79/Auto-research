#!/bin/bash
# scratch: sizes for the cleanup inventory (D44 (5)); read-only
cd "$(dirname "$0")/../../.." || exit 1
for d in temp bev_fusion/temp bev_fusion/runs bev_fusion/results bev_fusion/_to_delete bev_fusion/data/cache/gt bev_fusion/data/cache/radar bev_fusion/data/cache/img_front bev_fusion/data/cache/cam_feats bev_fusion/data/cache/cam_feats_mini bev_fusion/data/cache/small bev_fusion/data/cache/annotation_pts.npz bev_fusion/data/checkpoints bev_fusion/models data/autogt data/NuScenes bev_fusion_lane_center/runs bev_fusion_lane_center/temp bev_fusion_lane_center/data bev_fusion_lane_center/_to_delete bev_fusion_lane_center/models wsl; do
  [ -e "$d" ] && printf "%8s  %s\n" "$(du -shL "$d" 2>/dev/null | cut -f1)" "$d"
done
echo "--- runs: count and the biggest"
ls bev_fusion/runs | wc -l
du -s bev_fusion/runs/* 2>/dev/null | sort -rn | head -12 | awk '{printf "%8.1f GB  %s\n", $1/1048576, $2}'
echo "--- GT cache by signature (files)"
ls bev_fusion/data/cache/gt | sed -E 's/\.npz$//; s/.*_([0-9a-f]{10})$/\1/' | sort | uniq -c | sort -rn | head -12
