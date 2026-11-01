#!/usr/bin/env bash
# usage: commit_paths.sh <msgfile> <pathlist file> [<blobspec file: "path<TAB>source file" for partial files>]
# commits exactly the listed paths (working-tree state; deletions included) through a temporary index; HEAD moves only if unchanged meanwhile
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
msg=$1; list=$2; blobs=${3:-}
old=$(git rev-parse HEAD)
idx=$(mktemp -u /tmp/ar_idx_XXXX)
export GIT_INDEX_FILE=$idx
git read-tree HEAD
while IFS= read -r p; do [ -z "$p" ] && continue; git add -A -- "$p"; done < "$list"
if [ -n "$blobs" ]; then while IFS=$'\t' read -r p src; do h=$(git hash-object -w "$src"); git update-index --add --cacheinfo 100644,"$h","$p"; done < "$blobs"; fi
tree=$(git write-tree)
unset GIT_INDEX_FILE; rm -f "$idx"
c=$(git commit-tree "$tree" -p "$old" -F "$msg")
git update-ref HEAD "$c" "$old"
git diff --stat "$old" "$c" | tail -3
# sync the shared index for these paths (working tree untouched)
while IFS= read -r p; do [ -z "$p" ] && continue; git reset -q HEAD -- "$p" 2>/dev/null || true; done < "$list"
if [ -n "$blobs" ]; then while IFS=$'\t' read -r p src; do git reset -q HEAD -- "$p"; done < "$blobs"; fi
echo "committed $c"
