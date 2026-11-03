#!/bin/bash
# scratch: commit of the own text records (paths in commit_paths.txt) with commit_paths.py (temporary index; what other chats stage or edit stays out)
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t16/msg_final1.txt $(cat bev_fusion/temp/t16/commit_paths.txt | tr '\n' ' ') 2>&1 | tail -n 15
git log --oneline -3 | cut -c1-140
