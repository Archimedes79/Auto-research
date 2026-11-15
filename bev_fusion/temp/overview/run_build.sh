#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python temp/overview/build.py
