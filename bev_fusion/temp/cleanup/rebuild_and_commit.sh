#!/bin/bash
# scratch (Image Encoder, clean-base round): rebuild the blobs from the CURRENT HEAD and commit at once (another chat committed train.py between my last rebuild and the dry run)
cd /mnt/c/Auto-research || exit 1
bash bev_fusion/temp/cleanup/rebuild_blobs.sh 2>&1 | tail -12
echo "=== commit"
bash bev_fusion/temp/cleanup/commit_clean.sh 2>&1
echo "=== what the new commit changed (numstat against its parent):"
git show --numstat --format='%h %s' HEAD | cut -c1-170
