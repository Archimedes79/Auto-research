#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
/home/ml/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py runs/t16_full_s0 generalisation --n 1000 2>&1 | tail -60
