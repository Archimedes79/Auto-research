#!/bin/bash
# scratch: commit of the documentation cleanup (own paths, temporary index)
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t16/msg_docs.txt $(cat bev_fusion/temp/t16/commit_paths_docs.txt | tr '\n' ' ') 2>&1 | tail -n 6
git log --oneline -3 | cut -c1-130
git show --stat --format=%h HEAD | tail -n 3
