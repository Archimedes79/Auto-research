#!/bin/bash
# tracked bev_fusion/temp files and who points at them outside the run records (run.json, LEDGER.*, logs)
cd /mnt/c/Auto-research || exit 1
git ls-files bev_fusion/temp | grep -v '/README.md$' | while read -r f; do
  rel=${f#bev_fusion/}
  base=$(basename "$f")
  refs=$(git grep -l -F "$rel" -- . ':!bev_fusion/records/runs/*' ':!bev_fusion/records/LEDGER.*' ':!*/history/*' ':!bev_fusion/temp/*' 2>/dev/null | head -4 | tr '\n' ' ')
  if [ -z "$refs" ]; then refs=$(git grep -l -F "temp/${rel#temp/}" -- . ':!bev_fusion/records/runs/*' ':!bev_fusion/records/LEDGER.*' ':!*/history/*' ':!bev_fusion/temp/*' 2>/dev/null | head -4 | tr '\n' ' '); fi
  echo "$f | ${refs:-NONE}"
done
