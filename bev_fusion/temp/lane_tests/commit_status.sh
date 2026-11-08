#!/bin/bash
# scratch: commit the status file (own path only) with a message file
cd "$(dirname "$0")/../../.." || exit 1
python3 autoresearch/tools/check_git.py --paths bev_fusion/records/status_2026-11-07.md || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/lane_tests/commit_status_msg.txt bev_fusion/records/status_2026-11-07.md
