#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY bev_fusion/temp/cleanup/audit_b/dup_funcs.py $(git ls-files 'bev_fusion/tests/*.py' 'bev_fusion/studies/*.py' 'bev_fusion/temp/*.py' 'bev_fusion/training/*.py' 'bev_fusion/evaluation/*.py' 'bev_fusion/runner/*.py') 2>&1
