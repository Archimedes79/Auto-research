#!/bin/bash
cd "$(dirname "$0")/../.."
OUT=temp/landing/blocked_rows.md
: > $OUT
while IFS=$'\t' read -r d mb; do
  out=$(python3 ../autoresearch/tools/clean_temp.py "$d" 2>&1)
  first=$(echo "$out" | grep -m1 "BLOCKED:" | sed 's/^ *BLOCKED: //' | cut -c1-110)
  more=$(echo "$out" | grep -m1 -oE "and [0-9]+ more" | grep -oE "[0-9]+")
  n=$(echo "$out" | grep -c "BLOCKED:")
  tot=$((n + ${more:-0}))
  first=${first//|/\|}
  printf '| `temp/%s/` | %s MB | %s | `%s` |\n' "$d" "$mb" "$tot" "$first" >> $OUT
done < temp/landing/blocked.tsv
echo done >> $OUT
