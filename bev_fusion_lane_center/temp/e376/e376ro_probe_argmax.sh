#!/bin/bash
cd "$(dirname "$0")/../.."
for r in e373_cb_s0 e373_cb_s1; do env OMP_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh e376ro_probe_argmax.py $r 2>&1 | tail -4; done
