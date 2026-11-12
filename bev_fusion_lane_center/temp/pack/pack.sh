#!/bin/bash
# Copy the two reports and the data behind them into one folder (copy only, nothing is moved or deleted).
set -u
cd "$(dirname "$0")/../.."          # bev_fusion_lane_center
OUT=results/Spurmitte_Bericht_2026-11-12
mkdir -p "$OUT"

# 1 / 2: the reports as built (index.html + img/ + vid/)
mkdir -p "$OUT/1_Bericht_E2E" "$OUT/2_Bericht_Kreuzungen"
cp -r results/E2E.report/. "$OUT/1_Bericht_E2E/"
cp -r results/Junction.report/img results/Junction.report/vid results/Junction.report/index.html "$OUT/2_Bericht_Kreuzungen/"

# 3: data behind the numbers
D="$OUT/3_Daten"
mkdir -p "$D"
# 3a: small result files of every run of the junction thread (no checkpoints, no dumps)
for r in runs/e28[5-9]_* runs/e29[0-9]_* runs/e3[01][0-9]_* runs/e28[5-9] runs/e29[0-9] runs/e3[01][0-9]; do
  [ -d "$r" ] || continue
  case "$r" in *smoke*|*dummy*|*e315*) continue;; esac     # smoke tests are no results; E315 is still running
  find "$r" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name 'log.txt' -o -name '*.csv' -o -name '*.md' \) -size -2M \
    -exec cp --parents {} "$D/" \;
done
# 3b: the scripts' side files the reports read
mkdir -p "$D/temp_results"
cp temp/e309/*.json "$D/temp_results/" 2>/dev/null
cp temp/jr/model_cost.json "$D/temp_results/" 2>/dev/null
mkdir -p "$D/temp_results/snap"
cp temp/gt2/snap/*.json "$D/temp_results/snap/" 2>/dev/null
# 3c: per-frame dumps and weights of the final system E312 (so the numbers can be rescored)
for s in 0 1; do
  mkdir -p "$D/runs/e312_p_s$s"
  cp runs/e312_p_s$s/ends_val.pkl runs/e312_p_s$s/model.pt "$D/runs/e312_p_s$s/" 2>/dev/null
done
# 3d: rendered frames + videos of the final system
cp -r results/l6_view_E312 "$D/Bilder_Videos_E312"

# 4: experiment cards E285-E320 (question, rule, result), ledger, handover
mkdir -p "$OUT/4_Karten_Ledger"
for e in experiments/E28[5-9]_* experiments/E29[0-9]_* experiments/E3[0-2][0-9]_*; do
  [ -d "$e" ] || continue
  mkdir -p "$OUT/4_Karten_Ledger/$(basename "$e")"
  cp "$e"/experiment.md "$OUT/4_Karten_Ledger/$(basename "$e")/" 2>/dev/null
done
cp experiment_execution/LEDGER.csv "$OUT/4_Karten_Ledger/"
cp HANDOVER.md "$OUT/4_Karten_Ledger/"

# 5: the scripts that built the reports
mkdir -p "$OUT/5_Skripte_Bericht"
cp temp/jr/build_e2e_report.py temp/jr/build_junction_report.py temp/jr/report_text.py temp/jr/head.html "$OUT/5_Skripte_Bericht/" 2>/dev/null

echo done
du -sh "$OUT"
find "$OUT" -type f | wc -l
