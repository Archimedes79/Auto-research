#!/bin/bash
# after the landing commit: bring the working copies of the three files that carried foreign hunks in line with the new HEAD
cd /mnt/c/Auto-research || exit 1
S=bev_fusion/temp/cleanup
K=$S/foreign_working_copies
mkdir -p $K
old=539edcd6
# keep what was there (nothing is lost)
cp .gitignore $K/gitignore.working; cp bev_fusion/temp/README.md $K/temp_README.working; cp bev_fusion/tests/README.md $K/tests_README.working
# .gitignore: working copy (old HEAD + the foreign removal of `.jobs/`) merged with the new HEAD
git show $old:.gitignore > $K/gitignore.base
git show HEAD:.gitignore > $K/gitignore.new
cp .gitignore $K/gitignore.merged
git merge-file $K/gitignore.merged $K/gitignore.base $K/gitignore.new && echo "gitignore merged clean" && cp $K/gitignore.merged .gitignore
# temp/README.md (the rows the foreign hunk edited are gone) and tests/README.md (stale copy with a test_frozen row): the new HEAD
git show HEAD:bev_fusion/temp/README.md > bev_fusion/temp/README.md
git show HEAD:bev_fusion/tests/README.md > bev_fusion/tests/README.md
echo "--- tracked changes left in main:"
git status --porcelain | grep -v '^??'
echo "--- .gitignore diff vs HEAD:"; git diff -- .gitignore | grep '^[+-]' | grep -v '^+++\|^---'
