#!/usr/bin/env bash
# build the paper report (results/E2E_Champion_Report): ladder chart, html, number check. The parts are edited by hand now (no regeneration from the skeleton).
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
RP=results/E2E_Champion_Report
$PY /mnt/c/Auto-research/reporter/plots.py bars $RP/parts/figures/ladder.json $RP/parts/figures/ladder.png && \
$PY /mnt/c/Auto-research/reporter/reporter.py build $RP $1 && \
$PY /mnt/c/Auto-research/reporter/reporter.py check $RP $1
