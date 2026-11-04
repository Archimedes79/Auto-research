#!/bin/bash
# scratch: paths of the documentation-cleanup commit (D49): own edits, the two removed archive folders (every file by name), the new records; then check_git --paths
cd /mnt/c/Auto-research || exit 1
H=bev_fusion_lane_center/experiment_execution/history
{
  echo CLAUDE.md
  echo bev_fusion_lane_center/tools/README.md
  echo bev_fusion_lane_center/literature/OVERVIEW.md
  echo bev_fusion_lane_center/literature/README.md
  echo bev_fusion_lane_center/experiment_execution/DECISIONS.md
  echo bev_fusion/records/README.md
  echo bev_fusion/records/doc_review_2026-11-03.md
  echo bev_fusion/records/doc_review_2026-11-03_autoresearch.md
  echo bev_fusion/records/doc_review_2026-11-03_bev_fusion.md
  echo bev_fusion/records/doc_review_2026-11-03_lane.md
  git ls-tree -r --name-only HEAD -- $H/migration_tools $H/literature_sources
} > bev_fusion/temp/t16/commit_paths_docs.txt
wc -l bev_fusion/temp/t16/commit_paths_docs.txt
python3 autoresearch/tools/check_git.py --paths $(cat bev_fusion/temp/t16/commit_paths_docs.txt | tr '\n' ' ') 2>&1 | tail -n 6
