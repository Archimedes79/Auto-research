#!/bin/bash
# scratch (Image Encoder, clean-base round): dry run of the reverse patch of the B2 code commit 1648bcc24d, file by file, against the working tree (nothing is changed)
cd /mnt/c/Auto-research || exit 1
for f in bev_fusion/training/lane_branch.py bev_fusion/training/feature_cache.py bev_fusion/training/model.py bev_fusion/training/interface.py bev_fusion/training/startable.py \
         bev_fusion/training/train.py bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py bev_fusion/evaluation/bev_probe.py bev_fusion/evaluation/stream_drift.py \
         bev_fusion/tests/test_student_chain.py bev_fusion/tests/test_stream_tools.py bev_fusion/tests/test_student_encoder.py; do
  git show 1648bcc24d --format= -- $f > /tmp/b2_one.diff
  echo "== $f"
  patch -R --dry-run -p1 < /tmp/b2_one.diff 2>&1 | tail -4
done
