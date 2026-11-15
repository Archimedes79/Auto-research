#!/bin/bash
# data folder of the total report: the experiment table, the cards E273 - E319, the ledger and the small result files the report reads (copy only)
cd "$(dirname "$0")/../.."
OUT=results/Gesamtbericht_Spurmitte/daten
rm -rf "$OUT"; mkdir -p "$OUT/karten" "$OUT/ergebnisdateien/runs" "$OUT/ergebnisdateien/temp"
cp temp/jr/experiments_table.json temp/jr/cards_summary.json "$OUT/" 2>/dev/null
for e in experiments/E27[3-9]_* experiments/E2[89][0-9]_* experiments/E30[0-9]_* experiments/E31[0-9]_*; do
  [ -d "$e" ] || continue
  mkdir -p "$OUT/karten/$(basename "$e")"; cp "$e"/experiment.md "$OUT/karten/$(basename "$e")/"
done
cp experiment_execution/LEDGER.csv HANDOVER.md "$OUT/"
for r in runs/e284_h2_s? runs/e286_j1_s? runs/e287_p2_s? runs/e298_w_s? runs/e303_s_s? runs/e305_e_s? runs/e307_n_s? runs/e308_t_s? runs/e312_p_s? runs/e313_s_s? runs/e314_p_s? runs/e273/e273_p_s?; do
  [ -d "$r" ] || continue
  find "$r" -maxdepth 1 -type f \( -name 'ends_card*.json' -o -name 'maneuver_val.json' -o -name 'jmaps_val*.json' -o -name 'config.json' -o -name 'log.txt' -o -name 'timing.json' -o -name 'card.json' \) -exec cp --parents {} "$OUT/ergebnisdateien/" \;
done
mkdir -p "$OUT/ergebnisdateien/runs/eval/e312_test"; cp runs/eval/e312_test/card_s*.json runs/eval/e312_test/results.md "$OUT/ergebnisdateien/runs/eval/e312_test/" 2>/dev/null
cp temp/e309/*.json temp/jr/model_cost.json "$OUT/ergebnisdateien/temp/" 2>/dev/null
mkdir -p "$OUT/ergebnisdateien/temp/snap"; cp temp/gt2/snap/*.json "$OUT/ergebnisdateien/temp/snap/" 2>/dev/null
mkdir -p "$OUT/skripte"; cp temp/jr/build_total_report.py temp/jr/build_e2e_report.py temp/jr/make_single_total.py temp/jr/extract_cards.py temp/jr/head.html temp/jr/svgcharts.py temp/jr/run_total.sh temp/jr/pack_total.sh "$OUT/skripte/"
du -sh "$OUT"; find "$OUT" -type f | wc -l
