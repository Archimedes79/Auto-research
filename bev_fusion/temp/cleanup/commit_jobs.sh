#!/bin/bash
# scratch: commit the three jobs READMEs through a temporary index
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/records/jobs/README.md bev_fusion/records/jobs/running/README.md bev_fusion/records/jobs/evaluation/README.md"
git status --short -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_jobs_msg.txt $P || exit 1
git log --oneline -1 | cut -c1-140
git status --short -- $P | cat
