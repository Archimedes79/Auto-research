#!/bin/bash
cd /mnt/c/Auto-research || exit 1
B=bev_fusion/temp/cleanup/blobs
git log -1 --format='HEAD before: %h' | cat
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/jr_commit_msg.txt \
  autoresearch/jobrunner bev_fusion/records/jobs/JOB_LAYER_PROPOSAL.md \
  --blob autoresearch/README.md=$B/autoresearch_README.md --blob .gitignore=$B/gitignore || exit 1
echo "--- what went in"
git show --stat --format='%h %s' HEAD | cut -c1-150
echo "--- the shared files: lines changed by the commit"
git show HEAD --numstat -- autoresearch/README.md .gitignore | cat
