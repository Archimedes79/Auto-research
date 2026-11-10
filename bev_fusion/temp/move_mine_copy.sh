#!/bin/bash
# scratch: move MY scratch of temp/cleanup into temp/runner_round (the folder clean_temp.py then tags and moves); the files of the lead's round stay
cd /mnt/c/Auto-research/bev_fusion/temp || exit 1
mkdir -p runner_round
cd cleanup || exit 1
keep=" files.txt gates_push.txt selftest_push.txt check_git_full.txt dead_candidates.py dead_candidates.txt dead_candidates2.py dead_candidates2.txt dead_candidates3.py dead_candidates3.txt "
n=0
for f in *; do
  case "$keep" in *" $f "*) continue;; esac
  mv "$f" ../runner_round/ && n=$((n + 1))
done
echo "moved $n entries; left in cleanup:"; ls
