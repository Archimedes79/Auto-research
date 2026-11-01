#!/usr/bin/env bash
# removes ONLY the symlink cam_feats_bevfusion inside the verify copy (rm without -r on a symlink never touches its target), then checks the
# target on D: and the original link
set -eu
V=/mnt/c/Auto-research/bev_fusion/temp/radar/_scratch_patches/verify/bev_fusion/data/cache/cam_feats_bevfusion
O=/mnt/c/Auto-research/bev_fusion/data/cache/cam_feats_bevfusion
T=/mnt/d/Auto-research/bev_fusion_cache_bevfusion/cam_feats_bevfusion
echo "copy entry : $V"
if [ -L "$V" ]; then
  echo "  is a symlink -> $(readlink "$V")"
  rm "$V"
  echo "  removed the link"
else
  echo "  NOT a symlink here, nothing removed"
fi
[ -e "$V" ] && echo "  STILL THERE" || echo "  gone"
if [ -L "$O" ]; then echo "original link intact -> $(readlink "$O")"; else echo "ORIGINAL LINK MISSING"; fi
echo "D: target exists: $([ -d "$T" ] && echo yes || echo NO), first entries:"
ls -f "$T" 2>/dev/null | head -4
