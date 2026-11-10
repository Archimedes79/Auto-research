#!/bin/bash
# scratch (Image Encoder, clean-base round): the diff of my cleanup as it will be committed: the clean files against HEAD, the shared files as HEAD + my hunk (temp/cleanup/headcopy) against HEAD
# (evaluation/ files are not part of it: the Evaluation chat owns them and keeps the lane_img_old key)
cd /mnt/c/Auto-research || exit 1
H=bev_fusion/temp/cleanup/headcopy
OUT=bev_fusion/temp/cleanup/t22_cleanup.diff
CLEAN="bev_fusion/training/lane_branch.py bev_fusion/training/feature_cache.py bev_fusion/training/model.py bev_fusion/training/interface.py bev_fusion/tests/test_student_chain.py bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/training/configs/t22_B2_s0.json"
SHARED="bev_fusion/training/train.py bev_fusion/training/startable.py bev_fusion/training/README.md bev_fusion/training/configs/README.md bev_fusion/tests/README.md"
: > $OUT
git diff -- $CLEAN >> $OUT
for f in $SHARED; do
  git show HEAD:$f > /tmp/head_one.txt
  diff -u --label a/$f --label b/$f /tmp/head_one.txt $H/$f >> $OUT
done
git diff --stat -- $CLEAN | tail -9
for f in $SHARED; do
  git show HEAD:$f > /tmp/head_one.txt
  echo "$f: $(diff /tmp/head_one.txt $H/$f | grep -c '^[<>]') changed lines (HEAD copy vs HEAD)"
done
