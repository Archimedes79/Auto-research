#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY bev_fusion/temp/cleanup/audit_b/dead_funcs.py $(git ls-files 'bev_fusion/studies/*.py' 'bev_fusion/tests/*.py' 'bev_fusion/tests/support/*.py') 2>&1
