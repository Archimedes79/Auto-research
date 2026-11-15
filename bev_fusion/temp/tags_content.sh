#!/bin/bash
# scratch: for each local-only tag: the files that exist only in the tag (not in HEAD), and how many of them still exist on disk at the same path (read-only)
cd /mnt/c/Auto-research || exit 1
git ls-tree -r --name-only HEAD | sort > /tmp/head_files.txt
for t in autogt/scratch-2026-11-02 autogt/scratch-l5-2026-11-07 autogt/scratch-lane-2026-11-04 autogt/scratch-masks-2026-11-05 autogt/scratch-release-2026-11-04 autogt/scratch-v10-2026-11-05 lcr/e273-scratch lcr/e273-scratch-report lcr/e274-scratch; do
  git ls-tree -r --name-only "$t" | sort > /tmp/tag_files.txt
  only=$(comm -13 /tmp/head_files.txt /tmp/tag_files.txt)
  n=$(echo "$only" | grep -c .)
  ondisk=0; total=0
  while read f; do [ -z "$f" ] && continue; total=$((total+1)); [ -e "$f" ] && ondisk=$((ondisk+1)); done <<< "$only"
  roots=$(echo "$only" | sed 's#/[^/]*$##' | sort | uniq -c | sort -rn | head -3 | tr '\n' ';' | cut -c1-200)
  echo "$t: only in tag $n files, still on disk $ondisk | $roots"
done
echo "=== is the tag the only holder? (other refs containing those commits)"
for t in autogt/scratch-2026-11-02 lcr/e273-scratch; do echo "$t: branches containing: $(git branch -a --contains "$t" 2>/dev/null | wc -l)"; done
echo "=== where scratch folders went:"; ls /mnt/c/Auto-research/bev_fusion/_to_delete /mnt/c/Auto-research/bev_fusion_lane_center/_to_delete 2>/dev/null | head -20
