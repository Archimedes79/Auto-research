#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY bev_fusion/temp/cleanup/audit_b/test_overview.py 2>&1
