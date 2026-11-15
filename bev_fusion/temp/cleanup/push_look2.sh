#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py > /tmp/cg_mine.txt 2>&1; echo "exit=$?"
grep -v -E 'pushing the tags would publish' /tmp/cg_mine.txt | cut -c1-230 | head -n 40
echo "== tags lines count:"; grep -c 'pushing the tags would publish' /tmp/cg_mine.txt
echo "== unpushed tags:"; grep -iE 'tags? (are )?not on|not on origin|held back|safe to push' /tmp/cg_mine.txt | cut -c1-300 | head -n 6
