#!/usr/bin/env bash
# commit own paths through a temporary index (the shared index belongs to every chat), move main with a checked update-ref, refresh the shared index entries of those paths
# usage from the repository root: wsl bash bev_fusion/temp/pitch/commit_paths.sh <message file> <path> [<path> ...]
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MSG=$1; shift
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- "$@"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$MSG")
unset GIT_INDEX_FILE
git update-ref -m "own paths" refs/heads/main "$commit" "$old"
git reset -q HEAD -- "$@"
git log --oneline -1
git show --stat --format= "$commit" | tail -6
