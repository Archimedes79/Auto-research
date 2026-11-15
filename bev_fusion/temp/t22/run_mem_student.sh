#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
$HOME/miniconda3/envs/gcar/bin/python temp/t22/mem_student.py 2>&1 | tail -8
