#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
R=/mnt/c/Auto-research/bev_fusion_lane_center
OUT=$R/results/Ergebnis_E2E_Champion_2026-11-17/E2E_Champion_Bericht_SKELETON.html
BFLC_ROOT=$R BFLC_OUT=$OUT /home/ml/miniconda3/envs/gcar/bin/python temp/jr/build_champion.py
grep -o "PENDING" "$OUT" | wc -l
