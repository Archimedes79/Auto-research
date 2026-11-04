#!/bin/bash
# first-pass numbers check of every deck slide against the primary files and the reviewed brief (advisory): which decimal numbers of a slide exist in none of the sources
cd "$(dirname "$0")/../../.." || exit 1
S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/3277f09c-f1a5-422e-b9e7-06db879246c7/scratchpad/deck/project/slides
SRC="bev_fusion/records/LEDGER.csv bev_fusion_lane_center/experiment_execution/LEDGER.csv bev_fusion/temp/pitch/facts.json bev_fusion/temp/pitch/maps_lfl.json bev_fusion/temp/pitch/maps_lfl.md bev_fusion/temp/pitch/ladder.md
 bev_fusion/experiments/t10_dual_features/experiment.md bev_fusion/experiments/t10_dual_features/results_x50_2seeds.md bev_fusion/experiments/t3_improve_loop/report.md bev_fusion/experiments/t6_soa_gap/report.md
 bev_fusion_lane_center/reports/source/weekly/case_autoresearch_2026-11-01.md bev_fusion_lane_center/PROBLEM.md"
for f in bev_fusion/experiments/t7_lane_branch/*.md bev_fusion/experiments/t11_vel_cause/*.md bev_fusion/experiments/t11_vel_hurdle/*.md bev_fusion/experiments/t8_*/*.md bev_fusion/experiments/t4_*/*.md; do [ -f "$f" ] && SRC="$SRC $f"; done
total=0
for s in $S/*.html; do
  out=$(python3 autoresearch/tools/check_numbers.py "$s" $SRC --show 60 2>&1)
  echo "=== $(basename $s): $(echo "$out" | head -1)"
  echo "$out" | sed -n '2,70p' | cut -c1-200
done
