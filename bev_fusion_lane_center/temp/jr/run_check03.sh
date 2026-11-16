#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
P=results/Ergebnis_Spurmitte_2026-11-15/parts
S=""
for f in experiments/E*_*/experiment.md; do S="$S $f"; done
S="$S experiments/E356_node-spacing/e355_step0_results.md $P/03_numbers.json runs/eval/jtiming/e351_quiet_eager.json runs/eval/jtiming/e353_time_s0.json experiment_execution/LEDGER.csv"
python3 ../autoresearch/tools/check_numbers.py $P/03_results.md $S --exact --show 80 2>&1 | tail -60 | cut -c1-200
