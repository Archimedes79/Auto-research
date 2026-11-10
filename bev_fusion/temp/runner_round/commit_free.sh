#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/cleanup/add_rows.py || exit 1
P="bev_fusion/evaluation/bev_free.py bev_fusion/tests/test_bev_free.py"
git status --short -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P bev_fusion/evaluation/README.md bev_fusion/tests/README.md || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_free_msg.txt $P \
  --blob bev_fusion/evaluation/README.md=bev_fusion/temp/cleanup/blobs/bev_fusion_evaluation_README.md \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/cleanup/blobs/bev_fusion_tests_README.md || exit 1
git status --short -- bev_fusion/evaluation/README.md bev_fusion/tests/README.md | cat
echo "(a status line above for tests/README.md is another chat's hunk; my rows are in HEAD)"
