#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
bash tools/env/_py14.sh e309/sens.py ${1:-0}
