#!/bin/bash
cd /mnt/c/Auto-research || exit 1
B=bev_fusion/temp/cleanup/blobs
git show HEAD:autoresearch/README.md > /tmp/readme_head.md
git show HEAD:.gitignore > /tmp/gitignore_head
echo "README: $(diff /tmp/readme_head.md $B/autoresearch_README.md | grep -c '^[<>]') changed line(s)"; diff /tmp/readme_head.md $B/autoresearch_README.md | cut -c1-120
echo ".gitignore: $(diff /tmp/gitignore_head $B/gitignore | grep -c '^[<>]') changed line(s)"; diff /tmp/gitignore_head $B/gitignore
