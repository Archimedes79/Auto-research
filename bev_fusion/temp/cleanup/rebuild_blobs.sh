#!/bin/bash
# scratch (Image Encoder, clean-base round): rebuild the blobs of the shared files from the CURRENT HEAD right before the commit (HEAD + my hunk only):
# the four code files get the reverse patch of the B2 commit, the three READMEs get the regex edits (head only; the working files were edited already).
# The previous copies are kept in headcopy_prev and diffed, so a commit of another chat between the two builds shows up.
cd /mnt/c/Auto-research || exit 1
H=bev_fusion/temp/cleanup/headcopy
rm -rf ${H}_prev; cp -r $H ${H}_prev
rm -rf $H; mkdir -p $H
for f in bev_fusion/training/train.py bev_fusion/training/startable.py; do                       # evaluation/ files are the Evaluation chat's: not mine, not committed (the lead, 2026-11-10)
  mkdir -p $H/$(dirname $f)
  git show HEAD:$f > $H/$f
  git show 1648bcc24d --format= -- $f > /tmp/b2_one.diff
  patch -R -p1 -d $H --no-backup-if-mismatch < /tmp/b2_one.diff 2>&1 | tail -3
done
python3 bev_fusion/temp/cleanup/edit_readmes.py bev_fusion/temp/cleanup/edits_b2_readmes.py --head-only
python3 bev_fusion/temp/cleanup/edit_readmes.py bev_fusion/temp/cleanup/edits_tests_readme.py --head-only
echo "--- rebuilt copies against the previous ones (empty = HEAD did not move for these files):"
diff -r ${H}_prev $H && echo "identical"
echo "--- HEAD now:"; git log -1 --format='%h %ad %s' --date=format:%H:%M | cut -c1-150
