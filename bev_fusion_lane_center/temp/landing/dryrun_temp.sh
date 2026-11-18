#!/bin/bash
# clean_temp.py DRY RUN over every folder of temp/ (list only: nothing is moved or deleted). Output: temp/landing/dryrun_temp.tsv  (folder, MB, verdict, first reason)
cd "$(dirname "$0")/../.."
OUT=temp/landing/dryrun_temp.tsv
: > $OUT
for d in $(ls temp); do
  [ -d "temp/$d" ] || continue
  mb=$(du -sm "temp/$d" 2>/dev/null | cut -f1)
  out=$(python3 ../autoresearch/tools/clean_temp.py "$d" 2>&1); rc=$?
  first=$(echo "$out" | grep -iE "refus|would|blocked|imports|names|changed in the last|nothing" | head -1 | cut -c1-200)
  printf '%s\t%s\t%s\t%s\n' "$d" "$mb" "$rc" "$first" >> $OUT
done
echo done >> $OUT
