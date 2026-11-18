#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
nice -n 10 bash tools/env/_py14.sh test_token_time_ens.py > temp/e379/test_tte.out 2>&1
