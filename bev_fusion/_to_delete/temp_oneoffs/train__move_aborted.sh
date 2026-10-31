#!/usr/bin/env bash
# One-off: move an aborted run folder to _to_delete/ (nothing is deleted) and list what is there.
cd "$(dirname "$0")/../.." || exit 1
ps aux | grep -E "[t]1_r1a" | head -3
mkdir -p _to_delete
mv "runs/$1" "_to_delete/runs_$1_$2" && echo "moved runs/$1 -> _to_delete/runs_$1_$2"
ls _to_delete | head -20
