#!/usr/bin/env bash
# the evaluation code of HEAD (before the centre-line change) next to the working tree: old/bev_fusion/evaluation, old/autoresearch/evalsuite (+ the links it needs)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
rm -rf "$HERE/old"
mkdir -p "$HERE/old"
cd "$REPO"
git archive HEAD bev_fusion/evaluation bev_fusion/training bev_fusion/tests/support autoresearch/evalsuite autoresearch/tools autogt | tar -x -C "$HERE/old"
ln -s "$REPO/bev_fusion/data" "$HERE/old/bev_fusion/data"
ln -s "$REPO/bev_fusion/runs" "$HERE/old/bev_fusion/runs"
git rev-parse HEAD > "$HERE/old/HEAD.txt"
du -sh "$HERE/old"
cat "$HERE/old/HEAD.txt"
