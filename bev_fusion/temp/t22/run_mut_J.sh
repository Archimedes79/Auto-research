#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
$HOME/miniconda3/envs/gcar/bin/python temp/t22/mut_J.py > temp/t22/mut_J.log 2>&1
echo "exit $?" >> temp/t22/mut_J.log
