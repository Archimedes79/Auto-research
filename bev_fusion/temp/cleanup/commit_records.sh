#!/bin/bash
cd /mnt/c/Auto-research || exit 1
# .gitignore: HEAD + my two lines (the GT chat has an uncommitted rename hunk in the working copy)
git show HEAD:.gitignore | awk '{print} /^\/bev_fusion\/results\/\*\/bundle\/$/ {print "# the result reports and overview pages (results/BEV_*/: videos, html, the text the build leaves) are outputs the build tools make again, not records"; print "/bev_fusion/results/BEV_*/"}' > bev_fusion/temp/cleanup/gitignore_blob
echo "HEAD $(git show HEAD:.gitignore | wc -l) lines, blob $(wc -l < bev_fusion/temp/cleanup/gitignore_blob) lines"
git diff --no-index --numstat <(git show HEAD:.gitignore) bev_fusion/temp/cleanup/gitignore_blob | head -2
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/records_msg.txt --blob .gitignore=bev_fusion/temp/cleanup/gitignore_blob $(cat bev_fusion/temp/cleanup/untracked_records.txt | tr '\n' ' ') || exit 1
git show --stat --format='%h %s' HEAD | tail -3 | cut -c1-130
echo "still untracked in records/results:"; git status --porcelain -- bev_fusion/records bev_fusion/results | grep '^??' | cut -c1-120
