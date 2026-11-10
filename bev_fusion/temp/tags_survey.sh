#!/bin/bash
# scratch: which tags exist, which are only local, how old, and what they hold (read-only)
cd /mnt/c/Auto-research || exit 1
git ls-remote --tags origin 2>/dev/null | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_tags.txt
git tag | sort > /tmp/local_tags.txt
echo "local tags: $(wc -l < /tmp/local_tags.txt), on origin: $(wc -l < /tmp/remote_tags.txt)"
echo "=== only local (not on origin):"
comm -23 /tmp/local_tags.txt /tmp/remote_tags.txt | while read t; do
  d=$(git log -1 --format=%cs "$t" 2>/dev/null)
  n=$(git ls-tree -r --name-only "$t" 2>/dev/null | wc -l)
  big=$(git ls-tree -r -l "$t" 2>/dev/null | awk '$4>1000000{c++} END{print c+0}')
  echo "  $t  commit $d  files $n  >1MB: $big"
done
echo "=== tag families on origin (prefix -> count, oldest, newest):"
sed 's#/.*##' /tmp/remote_tags.txt | sort | uniq -c | sort -rn | head -12
echo "=== remote tags by prefix with dates (oldest 12):"
for t in $(cat /tmp/remote_tags.txt); do echo "$(git log -1 --format=%cs "$t" 2>/dev/null) $t"; done | sort | head -12
