#!/bin/bash
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python /mnt/c/Auto-research/bev_fusion/temp/overview/steps_scan.py 2>&1 | head -60
