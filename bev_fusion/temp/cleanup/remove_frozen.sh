#!/bin/bash
# FROZEN.md -> records/frame_changes.md (header rewritten, changelog verbatim); the hash gate goes
cd /mnt/c/Auto-research/bev_fusion || exit 1
set -e
{ cat temp/cleanup/frame_changes_head.md; sed -n '/^## Changelog/,$p' FROZEN.md; } > records/frame_changes.md
grep -c '^- 20' records/frame_changes.md
rm FROZEN.md frozen.json tests/test_frozen.py
ls records/frame_changes.md
