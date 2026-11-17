#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py > /tmp/cg.txt 2>&1
grep -v "pushing the tags would publish" /tmp/cg.txt | head -40
echo "--- lines about tags:"; grep -c "pushing the tags would publish" /tmp/cg.txt
grep "pushing the tags would publish" /tmp/cg.txt | sed 's/.*(commit \(.*\))/\1/' | sort | uniq -c | head
