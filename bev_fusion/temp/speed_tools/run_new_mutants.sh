#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)"
~/miniconda3/envs/gcar/bin/python temp/speed_tools/check_anchors.py
ONLY="n:" nice -n 19 ~/miniconda3/envs/gcar/bin/python temp/speed_tools/mutate_host.py 2>&1 | tail -40 | cut -c1-170
