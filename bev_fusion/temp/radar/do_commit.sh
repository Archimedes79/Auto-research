#!/bin/bash
# one-off: commit test_doppler.py + its README row through commit_paths.py (arg 1: --dry-run or nothing)
set -u
cd /mnt/c/Auto-research
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/radar/commit_msg.txt \
  --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" ${1:-} \
  bev_fusion/tests/test_doppler.py bev_fusion/tests/README.md
