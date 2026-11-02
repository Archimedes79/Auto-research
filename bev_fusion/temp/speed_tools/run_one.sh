#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python temp/speed_tools/one_mutant.py "$1" 2>&1 | tail -20
