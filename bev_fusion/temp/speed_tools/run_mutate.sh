#!/bin/bash
# nice 19: the lead's t16.small_s0 trains on this box (load 4 of 12 cores); the main thread of a training is its critical path
cd /mnt/c/Auto-research/bev_fusion || exit 1
nice -n 19 ~/miniconda3/envs/gcar/bin/python temp/speed_tools/mutate_host.py 2>&1 | tail -50 | cut -c1-170
