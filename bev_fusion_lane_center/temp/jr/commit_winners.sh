#!/usr/bin/env bash
# usage: commit_winners.sh --dry-run | (no argument = commit). Commits reporter/, winners/ and the .gitignore blob (HEAD + two whitelist lines) with commit_paths.py.
cd /mnt/c/Auto-research || exit 1
cp bev_fusion_lane_center/temp/jr/gitignore_blob.txt /tmp/gitignore_mine.txt
# the ignore check of git add reads the WORKING .gitignore: add the two whitelist lines there too (idempotent; other chats' edits of that file stay)
grep -q '^!/reporter/$' .gitignore || sed -i '/^!\/GtGenerator\/$/a !/reporter/\n!/winners/' .gitignore
cat > /tmp/msg_winners.txt <<'EOF'
reporter/ (report tool: paper-style html from markdown parts, numbers from result files, number check, pack; plots.py; winner.py) and winners/ (small source snapshots of winning systems with manifest, no weights); .gitignore whitelists both

Standalone tools with selftests; winners/e324s_lanegraph is the first winner folder (the inference package of E324 s, 19 files, 90 KB).
EOF
python3 autoresearch/tools/commit_paths.py -F /tmp/msg_winners.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" reporter winners --blob .gitignore=/tmp/gitignore_mine.txt $1
