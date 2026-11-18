#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 bash tools/env/_py14.sh temp/e380/chord_probe.py --n 873 --out temp/e380/chord_probe.json > temp/e380/chord_probe.out 2>&1
