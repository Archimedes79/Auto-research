#!/bin/bash
# scratch (Image Encoder, clean-base round): the Evaluation chat owns evaluation/ and keeps the lane_img_old key (stored runs with student_frames 'all' still emit it): put back the three hunks of the
# B2 commit that my reverse patch had taken out of bev_adapter.py, bev_modules.py and bev_probe.py in the working tree (forward patch = the exact inverse of the reverse patch).
cd /mnt/c/Auto-research || exit 1
for f in bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py bev_fusion/evaluation/bev_probe.py; do
  git show 1648bcc24d --format= -- $f > /tmp/b2_one.diff
  echo "== $f"
  patch -p1 --no-backup-if-mismatch < /tmp/b2_one.diff 2>&1 | tail -3
done
echo "--- numstat of the three files against HEAD (before my change: bev_adapter 26/19, bev_modules 1/1, bev_probe none)"
for f in bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py bev_fusion/evaluation/bev_probe.py; do echo "$f: $(git diff --numstat -- $f | cut -f1,2)"; done
grep -c "lane_img_old" bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py bev_fusion/evaluation/bev_probe.py
find bev_fusion -name '*.rej' -o -name '*.orig' | head
