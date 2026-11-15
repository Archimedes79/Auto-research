#!/bin/bash
# scratch: bundle the 9 local-only tags (the only copy of their scratch files) into _to_delete, verify the bundle, then delete exactly those 9 local tags
cd /mnt/c/Auto-research || exit 1
TAGS="autogt/scratch-2026-11-02 autogt/scratch-l5-2026-11-07 autogt/scratch-lane-2026-11-04 autogt/scratch-masks-2026-11-05 autogt/scratch-release-2026-11-04 autogt/scratch-v10-2026-11-05 lcr/e273-scratch lcr/e273-scratch-report lcr/e274-scratch"
# only tags that are really not on origin
git ls-remote --tags origin | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_tags.txt
for t in $TAGS; do grep -qx "$t" /tmp/remote_tags.txt && { echo "STOP: $t is on origin"; exit 1; }; git rev-parse -q --verify "refs/tags/$t" >/dev/null || { echo "STOP: no tag $t"; exit 1; }; done
D=bev_fusion/_to_delete/old_tags_2026-11-10
mkdir -p $D
git bundle create $D/scratch_tags_9.bundle $(for t in $TAGS; do echo refs/tags/$t; done) 2>&1 | tail -2
git bundle verify $D/scratch_tags_9.bundle 2>&1 | tail -2
ls -la $D/scratch_tags_9.bundle | awk '{print $5, $9}'
# the bundle must contain all nine tags
n=$(git bundle list-heads $D/scratch_tags_9.bundle | grep -c "refs/tags/")
echo "tags in the bundle: $n"
[ "$n" = "9" ] || { echo "bundle incomplete: tags kept"; exit 1; }
# every object of the nine tags is in the bundle: a clone of the bundle has the same file counts
rm -rf /tmp/bundle_check && git clone -q $D/scratch_tags_9.bundle /tmp/bundle_check 2>&1 | tail -1
for t in autogt/scratch-2026-11-02 lcr/e273-scratch; do echo "$t: tag $(git ls-tree -r --name-only $t | wc -l) files, bundle clone $(git -C /tmp/bundle_check ls-tree -r --name-only refs/tags/$t | wc -l) files"; done
rm -rf /tmp/bundle_check
for t in $TAGS; do git tag -d "$t" | cut -c1-100; done
echo "local tags now: $(git tag | wc -l), not on origin: $(comm -23 <(git tag | sort) /tmp/remote_tags.txt | wc -l)"
