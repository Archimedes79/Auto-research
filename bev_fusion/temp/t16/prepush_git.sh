#!/bin/bash
# scratch: full check_git result before the push, with the known foreign tag (15f1f8da) hidden, and the errors that belong to commits of main
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py > /tmp/cg_full.txt 2>&1
echo "lines: $(wc -l < /tmp/cg_full.txt)"
grep -n -i -E "error|forbidden|scratch file|refus|ADD" /tmp/cg_full.txt | grep -v "15f1f8da" | head -20
echo "--- tail:"
tail -n 6 /tmp/cg_full.txt | cut -c1-200
echo "--- errors that name a commit of main (not the foreign tag):"
grep -c "pushing the tags would publish" /tmp/cg_full.txt
grep "pushing the tags would publish" /tmp/cg_full.txt | grep -v "15f1f8da" | head -5
echo "--- local tags not on origin:"
git for-each-ref refs/tags --format="%(refname:short)" | while read t; do git ls-remote --exit-code --tags origin "$t" > /dev/null 2>&1 || echo "$t"; done 2>/dev/null | head -20
