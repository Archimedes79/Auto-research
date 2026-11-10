#!/bin/bash
# scratch (Image Encoder, clean-base round 2026-11-10): reverse the B2 code commit 1648bcc24d (student_frames 'all') in the files where all of its hunks belong to 'all'.
# For the files that carry other chats' uncommitted hunks (train.py, startable.py, bev_adapter.py, bev_modules.py) the same reverse patch is also applied to a copy of their HEAD version
# (temp/cleanup/headcopy/...), which becomes the blob that is committed (HEAD + my hunk only).
cd /mnt/c/Auto-research || exit 1
H=bev_fusion/temp/cleanup/headcopy
rm -rf $H; mkdir -p $H
CLEAN="bev_fusion/training/lane_branch.py bev_fusion/training/feature_cache.py bev_fusion/training/model.py bev_fusion/training/interface.py bev_fusion/evaluation/bev_probe.py"
SHARED="bev_fusion/training/train.py bev_fusion/training/startable.py bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py"
for f in $CLEAN $SHARED; do
  git show 1648bcc24d --format= -- $f > /tmp/b2_one.diff
  echo "== working tree: $f"
  patch -R -p1 < /tmp/b2_one.diff 2>&1 | tail -6
done
for f in $SHARED; do
  mkdir -p $H/$(dirname $f)
  git show HEAD:$f > $H/$f
  git show 1648bcc24d --format= -- $f > /tmp/b2_one.diff
  echo "== HEAD copy: $f"
  patch -R -p1 -d $H < /tmp/b2_one.diff 2>&1 | tail -6
done
echo "--- leftovers"
find bev_fusion -name '*.rej' -o -name '*.orig' | head
