#!/usr/bin/env bash
# One-off (2026-10-25): md5 of N random manifest files in the source and in the copy must agree.
# Usage: sample_checksums.sh <manifest.tsv> <source dir> <copy dir> [N=200]
set -euo pipefail
man="$1"; src="$2"; dst="$3"; n="${4:-200}"
bad=0
while IFS=$'\t' read -r p _; do
  a=$(md5sum < "$src/$p" | cut -d' ' -f1); b=$(md5sum < "$dst/$p" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "MD5 MISMATCH $p"; bad=$((bad + 1)); }
done < <(shuf -n "$n" "$man")
echo "checked $n random files, mismatches: $bad"
[ $bad = 0 ]
