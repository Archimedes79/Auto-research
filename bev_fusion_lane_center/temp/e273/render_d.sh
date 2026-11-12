#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
nice -n 12 bash tools/env/_py14.sh render_d.py scenes 11 39 23 27 17 15 2>&1 | tail -8
