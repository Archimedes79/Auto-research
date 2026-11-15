#!/bin/bash
# scratch (Image Encoder, clean-base round): the frame_changes.md line as a blob from the CURRENT HEAD (+ the same line in the working file, which may carry other chats' edits); one commit, own path only
cd /mnt/c/Auto-research || exit 1
P=bev_fusion/records/frame_changes.md
H=bev_fusion/temp/cleanup/headcopy
rm -f $H/$P
python3 bev_fusion/temp/cleanup/edit_readmes.py bev_fusion/temp/cleanup/edits_frame_changes.py --head-only || exit 1
git show HEAD:$P | wc -l | sed 's/^/HEAD file lines: /'; wc -l < $H/$P | sed 's/^/blob lines: /'
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -4
EXTRA=""; [ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_msg_frame.txt --blob $P=$H/$P $EXTRA 2>&1 | tail -6
git log --oneline -2 | cut -c1-150
git show --numstat --format= HEAD -- $P | cat
