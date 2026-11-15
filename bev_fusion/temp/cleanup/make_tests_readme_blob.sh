#!/bin/bash
# tests/README.md: the committed text from HEAD minus the test_frozen row (other chats' uncommitted hunks in the working copy stay out)
cd /mnt/c/Auto-research || exit 1
git show HEAD:bev_fusion/tests/README.md | grep -v '^| `test_frozen.py` |' > bev_fusion/temp/cleanup/tests_README_blob.md
echo "HEAD lines: $(git show HEAD:bev_fusion/tests/README.md | wc -l), blob lines: $(wc -l < bev_fusion/temp/cleanup/tests_README_blob.md)"
git diff --no-index --stat <(git show HEAD:bev_fusion/tests/README.md) bev_fusion/temp/cleanup/tests_README_blob.md | tail -1
