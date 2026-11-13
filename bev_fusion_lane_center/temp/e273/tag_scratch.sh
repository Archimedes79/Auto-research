#!/bin/bash
# one-off: keep the scratch of E273 (text files only) in a git tag lcr/e273-scratch through a temporary index (main is not touched); nothing is deleted or moved
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e273_scratch_index"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
{
  find $L/temp/e273 $L/temp/e273_probes $L/temp/e273_raster $L/temp/e273_scorer $L/temp/e273_view -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.txt' -o -name '*.log' \) -size -300k
  grep -lE "E273|e273|qgraph|qlinks|qnodes|qtargets|qraster|map_sim" $L/temp/*.py $L/temp/*.sh $L/temp/*.md 2>/dev/null || true
  find $L/temp -maxdepth 1 -name 'fix_*.py' -newermt 2026-11-04
} | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E273 scratch (text files of temp/e273*, the fix scripts and the launchers; scripts that made the pictures, the package files and the ledger notes; main is not touched)")
git tag lcr/e273-scratch "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -3
