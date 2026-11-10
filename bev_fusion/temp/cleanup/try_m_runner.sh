#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY -m runner methods | cut -c1-200
echo "exit $?"
$PY -m runner Runner.list-results runs/t20_base_s0 --json | head -5
echo "exit $?"
$PY -m runner Runner.predict runs/nope-xyz val; echo "exit $?"
