#!/bin/bash
# scratch: bundle all lcr tags, verify the bundle, delete them on origin (batches) and locally, verify. Stops at the first problem BEFORE any deletion.
cd /mnt/c/Auto-research || exit 1
git ls-remote --tags origin 2>/dev/null | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_tags.txt
git tag | grep '^lcr/' | sort > /tmp/lcr_local.txt
grep '^lcr/' /tmp/remote_tags.txt > /tmp/lcr_remote.txt
n=$(wc -l < /tmp/lcr_local.txt)
[ "$n" = "127" ] && [ "$(wc -l < /tmp/lcr_remote.txt)" = "127" ] || { echo "STOP: expected 127 lcr tags locally and on origin, found $n / $(wc -l < /tmp/lcr_remote.txt)"; exit 1; }
D=bev_fusion_lane_center/_to_delete/old_tags_2026-11-10
mkdir -p $D
git bundle create $D/lcr_tags_127.bundle $(sed 's#^#refs/tags/#' /tmp/lcr_local.txt) 2>&1 | tail -1
git bundle verify $D/lcr_tags_127.bundle 2>&1 | tail -1
ls -la $D/lcr_tags_127.bundle | awk '{print "bundle bytes:", $5}'
b=$(git bundle list-heads $D/lcr_tags_127.bundle | grep -c "refs/tags/lcr/")
echo "tags in the bundle: $b"
[ "$b" = "127" ] || { echo "STOP: bundle incomplete, nothing deleted"; exit 1; }
rm -rf /tmp/lcr_check && git clone -q $D/lcr_tags_127.bundle /tmp/lcr_check 2>&1 | tail -1
for t in lcr/E234.tr0 lcr/loop22-scratch lcr/pre-restructure-2026-11-03; do echo "$t: tag $(git ls-tree -r --name-only "$t" | wc -l) files, bundle clone $(git -C /tmp/lcr_check ls-tree -r --name-only refs/tags/$t | wc -l) files"; done
rm -rf /tmp/lcr_check
echo "=== deleting on origin"
split -l 40 /tmp/lcr_remote.txt /tmp/lcr_batch_
for f in /tmp/lcr_batch_*; do
  git push origin $(sed 's#^#:refs/tags/#' "$f" | tr '\n' ' ') 2>&1 | grep -c "\[deleted\]" | sed 's#^#deleted in this batch: #'
done
rm -f /tmp/lcr_batch_*
echo "=== deleting locally"
for t in $(cat /tmp/lcr_local.txt); do git tag -d "$t" >/dev/null; done
echo "=== verify"
git ls-remote --tags origin 2>/dev/null | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_after.txt
echo "lcr on origin now: $(grep -c '^lcr/' /tmp/remote_after.txt), local: $(git tag | grep -c '^lcr/'); other tags on origin: $(grep -vc '^lcr/' /tmp/remote_after.txt) (was $(grep -vc '^lcr/' /tmp/remote_tags.txt)), local other: $(git tag | grep -vc '^lcr/')"
