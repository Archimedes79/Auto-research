#!/bin/bash
cd /mnt/c/Auto-research || exit 1
B=bev_fusion
python3 autoresearch/tools/check_git.py --paths $B/records/frame_changes.md $B/FROZEN.md $B/frozen.json $B/tests/test_frozen.py $B/tests/run_gates.sh $B/README.md $B/project.json $B/records/README.md 2>&1 | tail -4
python3 autoresearch/tools/commit_paths.py -F $B/temp/cleanup/frozen_msg.txt \
  $B/records/frame_changes.md $B/FROZEN.md $B/frozen.json $B/tests/test_frozen.py $B/tests/run_gates.sh $B/README.md $B/project.json $B/records/README.md \
  --blob $B/tests/README.md=$B/temp/cleanup/tests_README_blob.md || exit 1
git show --stat --format='%h %s' HEAD | cut -c1-130
# bring the working copy of tests/README.md in line only for my hunk: the other chats' hunks stay as they are
