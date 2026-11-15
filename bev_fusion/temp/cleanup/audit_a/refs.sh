#!/bin/bash
# usage: refs.sh NAME [NAME...]  -> files (and counts) that mention NAME as a word, outside temp/ scratch
cd /c/Auto-research
for n in "$@"; do
  echo "=== $n"
  git grep -c -w -- "$n" -- . ':!bev_fusion/temp' ':!temp' ':!autoresearch/history' | sed 's/^/   /' | head -25
done
