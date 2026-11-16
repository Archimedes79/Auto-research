#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
R=/mnt/c/Auto-research/bev_fusion_lane_center
OUT=$R/results/Ergebnis_Spurmitte_2026-11-15/Spurmitte_Ergebnisbericht_2026-11-16e.html
ZIP=$R/results/Ergebnis_Spurmitte_2026-11-15/Spurmitte_Paket_2026-11-16e.zip
BFLC_ROOT=$R BFLC_OUT=$OUT /home/ml/miniconda3/envs/gcar/bin/python temp/jr/build_ergebnis.py || exit 1
python3 temp/jr/pack_paket.py --html "$OUT" --out "$ZIP" || exit 1
ls -l "$OUT" "$ZIP"
