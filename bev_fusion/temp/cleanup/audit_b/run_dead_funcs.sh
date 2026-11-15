#!/usr/bin/env bash
# usage: run_dead_funcs.sh file.py ...  (paths relative to the repo root); needs /tmp/audit_b_index.pkl (made by run_symbol_use.sh)
cd /mnt/c/Auto-research || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY bev_fusion/temp/cleanup/audit_b/dead_funcs.py "$@" 2>&1
