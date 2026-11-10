#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
P=~/miniconda3/envs/gcar/bin/python
du -sh runs/t20_base_s0/results/*
echo "--- list"; $P runner/bev_results.py list
echo "--- get --verify"; $P runner/bev_results.py get 7534 --verify; echo "exit $?"
echo "--- get --json (head)"; $P runner/bev_results.py get 7534 --json | head -24
echo "--- get nope"; $P runner/bev_results.py get nope; echo "exit $?"
echo "--- frames dir"; ls runs/t20_base_s0/results/*/frames
