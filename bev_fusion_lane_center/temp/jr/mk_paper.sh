#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
RP=results/E2E_Champion_Report
BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center python3 temp/jr/make_paper_report.py && \
$PY /mnt/c/Auto-research/reporter/plots.py bars $RP/parts/figures/ladder.json $RP/parts/figures/ladder.png && \
$PY /mnt/c/Auto-research/reporter/reporter.py build $RP && \
$PY /mnt/c/Auto-research/reporter/reporter.py check $RP
