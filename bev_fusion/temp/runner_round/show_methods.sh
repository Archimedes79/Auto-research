#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
PY=~/miniconda3/envs/gcar/bin/python
AR_PROJECT=../bev_fusion $PY -m evalsuite methods | cut -c1-230
echo ----
AR_PROJECT=../bev_fusion $PY -m trainsuite methods | cut -c1-200
