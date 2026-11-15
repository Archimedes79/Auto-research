#!/usr/bin/env bash
# usage: run_symbol_use.sh module.py ...   (paths relative to repo root)
cd /mnt/c/Auto-research || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY bev_fusion/temp/cleanup/audit_b/symbol_use.py "$@" 2>&1
