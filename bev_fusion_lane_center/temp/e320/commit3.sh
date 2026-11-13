#!/bin/bash
cd /mnt/c/Auto-research
B=bev_fusion_lane_center
P="$B/experiments/E320_soft-attach-replaces-snap/experiment.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 2
python3 autoresearch/tools/commit_paths.py -m "E320 card: the start is Friday 2026-11-13 08:30 after E316 (E315 pair stage 2.3 s/step, day chain 11 done about 06:15), result about 11:00 - 11:30" --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P 2>&1 | tail -n 4
