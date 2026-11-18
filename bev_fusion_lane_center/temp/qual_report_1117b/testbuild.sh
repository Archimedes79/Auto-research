#!/bin/bash
# scratch: test build + check of a copy of the report folder (results/E2E_Champion_Report_tb), the real folder is not rebuilt
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
RP=results/E2E_Champion_Report_tb
$PY /mnt/c/Auto-research/reporter/reporter.py build $RP && $PY /mnt/c/Auto-research/reporter/reporter.py check $RP
echo "exit $?"
