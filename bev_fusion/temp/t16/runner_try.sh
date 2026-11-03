#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
rm -rf temp/t16/runner_try
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 /home/ml/miniconda3/envs/gcar/bin/python runner/bev_runner.py full runs/t16_full_s0 --scene scene-0003 --first 3 --out temp/t16/runner_try 2>&1 | tail -25
ls -la temp/t16/runner_try | head
