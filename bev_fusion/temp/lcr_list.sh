#!/bin/bash
# scratch: the lcr tags, local and on origin (read-only)
cd /mnt/c/Auto-research || exit 1
git ls-remote --tags origin 2>/dev/null | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_tags.txt
git tag | sort > /tmp/local_tags.txt
grep '^lcr/' /tmp/remote_tags.txt > /tmp/lcr_remote.txt
grep '^lcr/' /tmp/local_tags.txt > /tmp/lcr_local.txt
echo "lcr on origin: $(wc -l < /tmp/lcr_remote.txt), local: $(wc -l < /tmp/lcr_local.txt), only local: $(comm -23 /tmp/lcr_local.txt /tmp/lcr_remote.txt | wc -l), only on origin: $(comm -13 /tmp/lcr_local.txt /tmp/lcr_remote.txt | wc -l)"
echo "non-lcr tags (must stay): $(grep -vc '^lcr/' /tmp/local_tags.txt)"
echo "sample:"; head -3 /tmp/lcr_remote.txt; tail -3 /tmp/lcr_remote.txt
echo "are any lcr tags reachable from main (would stay reachable)? $(for t in $(cat /tmp/lcr_local.txt); do git merge-base --is-ancestor "$t" origin/main 2>/dev/null && echo y; done | wc -l)"
echo "do docs in main name lcr/ tags? $(git grep -l 'lcr/[A-Za-z0-9]' -- '*.md' '*.csv' '*.json' | wc -l) files (sample: $(git grep -l 'lcr/[A-Za-z0-9]' -- '*.md' '*.csv' '*.json' | head -3 | tr '\n' ' '))"
