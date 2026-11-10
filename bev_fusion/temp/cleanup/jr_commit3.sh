#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git status --short autoresearch/jobrunner
P="autoresearch/jobrunner/api.py autoresearch/jobrunner/catalogue.py autoresearch/jobrunner/selftest.py autoresearch/jobrunner/README.md"
git diff --stat -- $P | cat
cd autoresearch && python3 -m jobrunner selftest 2>&1 | tail -n 1; cd ..
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/jr_commit3_msg.txt $P || exit 1
git show --stat --format='%h %s' HEAD | cut -c1-130
