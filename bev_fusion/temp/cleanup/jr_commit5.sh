#!/bin/bash
cd /mnt/c/Auto-research || exit 1
echo "toplevel: $(git rev-parse --show-toplevel)"
git status --short autoresearch/jobrunner
P="autoresearch/jobrunner/api.py autoresearch/jobrunner/catalogue.py autoresearch/jobrunner/call.py autoresearch/jobrunner/selftest.py autoresearch/jobrunner/README.md"
git diff --stat -- $P | cat
export TMPDIR=/tmp/jr_mine_c5; rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"
(cd autoresearch && python3 -m jobrunner selftest 2>&1 | tail -n 1)
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/jr_commit5_msg.txt $P || exit 1
git show --stat --format='%h %s' HEAD | cut -c1-130
echo "--- still uncommitted in jobrunner:"; git status --short autoresearch/jobrunner
