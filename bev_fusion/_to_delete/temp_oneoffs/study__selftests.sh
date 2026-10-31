#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY training/blocks_new.py && $PY training/model.py
