#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/records/LEDGER.csv bev_fusion/_to_delete/README.md bev_fusion/records/runs/chain.gt bev_fusion/records/runs/chain.predict bev_fusion/records/runs/chain.predict2 bev_fusion/records/runs/chain.score bev_fusion/records/runs/chain.t20 bev_fusion/records/runs/chain2.gt bev_fusion/records/runs/chain2.predict bev_fusion/records/runs/chain2.score bev_fusion/records/runs/chain2.t20"
git diff --stat -- bev_fusion/records/LEDGER.csv bev_fusion/_to_delete/README.md | cat
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 3 | cut -c1-200
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/records_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
git show --stat --format= HEAD | tail -n 4 | cut -c1-120
