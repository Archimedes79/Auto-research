#!/bin/bash
# scratch: removes the two leftover test folders of this chat (D63: Mirko's yes to deleting the test folders); lists them first; they hold stub checkpoints and images of test_student_chain.py only
cd /mnt/c/Auto-research/bev_fusion/temp || exit 1
for d in student_chain_*; do
  [ -d "$d" ] || continue
  echo "$d  $(du -sh "$d" | cut -f1)  $(find "$d" -type f | wc -l) files  newest: $(find "$d" -type f -printf '%TH:%TM\n' | sort | tail -1)"
done
for d in student_chain_*; do
  [ -d "$d" ] || continue
  case "$d" in student_chain_????????|student_chain_*_) rm -rf -- "$d" && echo "removed $d";; *) echo "skipped $d";; esac
done
ls -d student_chain_* 2>/dev/null | wc -l
