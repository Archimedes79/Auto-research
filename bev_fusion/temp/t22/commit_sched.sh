#!/bin/bash
# scratch: commit of my one schedule row (frozen.update -> done)
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -m "schedule: frozen.update done (FROZEN.md line + frozen.json committed, test_frozen green; t22.refC_gate next, armed for the night window)" \
  --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" bev_fusion/records/schedule.csv 2>&1 | tail -4
