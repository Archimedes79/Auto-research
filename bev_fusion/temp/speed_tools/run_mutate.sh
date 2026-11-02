#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
~/miniconda3/envs/gcar/bin/python temp/speed_tools/mutate_host.py 2>&1 | tail -30 | cut -c1-170
