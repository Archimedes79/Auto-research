#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
P=results/Ergebnis_Spurmitte_2026-11-15/parts
S=""
for e in E343 E345 E348 E334 E341 E324 E312 E351 E352 E353 E354 E355 E357 E273 E320 E338 E325 E317 E307 E305 E311 E286 E287 E275 E347 E291 E299 E319 E318; do
  for f in experiments/${e}_*/experiment.md; do [ -f "$f" ] && S="$S $f"; done
done
S="$S experiments/E363_e2e-stack-on-faithful-graphs/experiment.md experiments/E358_wiring-head-on-attached-ends/experiment.md experiments/E360_matrix-nms-in-the-e2e-stack/experiment.md experiments/E359_node-type-head-per-source-softmax/experiment.md experiments/E361_node-type-head-10000-steps/experiment.md experiments/E362_node-type-head-16000-steps/experiment.md experiments/E357_oracle-nodes-one-pair-head/experiment.md experiments/E2E_gap_table.md experiments/E292_matrix-nms-layer/experiment.md experiments/E350_wiring-head-on-e324s/experiment.md experiments/E356_node-spacing/e355_step0_results.md $P/03_numbers.json $P/03_results.md runs/eval/jtiming/e351_quiet_eager.json runs/eval/jtiming/e353_time_s0.json experiment_execution/LEDGER.csv"
for d in 00a_start 00_summary 08_open 09_rebuild; do
  echo "=== $d"
  python3 ../autoresearch/tools/check_numbers.py $P/$d.md $S --exact --show 80 2>&1 | tail -45
done
