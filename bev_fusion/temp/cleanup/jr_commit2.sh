#!/bin/bash
cd /mnt/c/Auto-research || exit 1
echo "toplevel: $(git rev-parse --show-toplevel)  HEAD: $(git log -1 --format=%h)"
cd autoresearch && python3 -m jobrunner selftest > /tmp/jr_final.txt 2>&1; rc=$?; tail -n 1 /tmp/jr_final.txt | cut -c1-120; cd ..
[ $rc -eq 0 ] || { echo "selftest failed, nothing committed"; tail -n 12 /tmp/jr_final.txt; exit 1; }
P="autoresearch/jobrunner/README.md autoresearch/jobrunner/__main__.py autoresearch/jobrunner/api.py autoresearch/jobrunner/selftest.py"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 2
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/jr_commit2_msg.txt $P || exit 1
git show --stat --format='%h %s' HEAD | cut -c1-130
echo "--- still uncommitted in jobrunner (not mine):"; git status --short autoresearch/jobrunner
