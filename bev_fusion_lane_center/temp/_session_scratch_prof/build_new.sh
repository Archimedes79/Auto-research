#!/bin/bash
# scratch: build + number check of the report folder given as $1 (results/<folder>)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
$PY /mnt/c/Auto-research/reporter/reporter.py build $1 && $PY /mnt/c/Auto-research/reporter/reporter.py check $1
echo "exit $?"
