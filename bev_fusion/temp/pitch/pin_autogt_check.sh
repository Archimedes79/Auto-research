#!/bin/bash
# is the live autogt (repository root) byte-identical to the one of the run snapshot d14597e2e8 (bev_fusion/autogt then), file by file? then a pinned copy for the resume of t10_dual_long_s0
cd "$(dirname "$0")/../../.." || exit 1
snap=d14597e2e8
echo "signature files (snapshot blob vs live file):"
for f in grid layers flow nuscenes_adapter nuscenes_gt; do
  a=$(git rev-parse "$snap:bev_fusion/autogt/$f.py" 2>/dev/null | cut -c1-12)
  b=$(git hash-object "autogt/$f.py" | cut -c1-12)
  [ "$a" = "$b" ] && s=same || s=DIFFERENT
  echo "  $f.py  snapshot $a  live $b  $s"
done
echo "other files of the snapshot's autogt package:"
git ls-tree -r --name-only "$snap" -- bev_fusion/autogt | sed 's#bev_fusion/autogt/##' | tr '\n' ' ' | fold -w 200
echo
echo "files of the live autogt package:"
ls autogt | tr '\n' ' ' | fold -w 200
echo
