#!/bin/bash
# the exact list of clean tags to push: check_git's clean line (cut BEFORE the held-back part), not on origin, not of a run that is still running
cd /mnt/c/Auto-research || exit 1
grep '^tags without a forbidden file' /tmp/cg3.txt | sed 's/^[^:]*: //; s/ | held back.*//' | tr ' ' '\n' | grep -v '^$' | sort -u > /tmp/clean_tags.txt
git ls-remote --tags origin 2>/dev/null | awk '{print $2}' | sed 's#^refs/tags/##; s#\^{}$##' | sort -u > /tmp/remote_tags2.txt
git tag | sort > /tmp/local_tags2.txt
comm -23 /tmp/local_tags2.txt /tmp/remote_tags2.txt > /tmp/missing2.txt
comm -12 /tmp/clean_tags.txt /tmp/missing2.txt > /tmp/cand.txt
ps -eo cmd | grep -o 'run.py .*--id [A-Za-z0-9_.]*' | sed 's/.*--id //' | sort -u > /tmp/running_ids.txt
cp /tmp/cand.txt /tmp/push_tags3.txt
while read -r id; do [ -n "$id" ] && grep -v -x -e "lcr/$id" -e "bf/$id" /tmp/push_tags3.txt > /tmp/push_tags3.tmp && mv /tmp/push_tags3.tmp /tmp/push_tags3.txt; done < /tmp/running_ids.txt
grep -v -e '^lcr/E353\.' -e '^lcr/E354\.' /tmp/push_tags3.txt > /tmp/push_tags3.tmp && mv /tmp/push_tags3.tmp /tmp/push_tags3.txt
echo "clean: $(wc -l < /tmp/clean_tags.txt) missing on origin: $(wc -l < /tmp/missing2.txt) candidates: $(wc -l < /tmp/cand.txt) to push: $(wc -l < /tmp/push_tags3.txt)"
echo "missing but NOT pushed:"; comm -23 /tmp/missing2.txt /tmp/push_tags3.txt | tr '\n' ' '; echo
echo "running ids: $(tr '\n' ' ' < /tmp/running_ids.txt)"
