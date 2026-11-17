#!/bin/bash
# scratch (2026-11-17): graphs of the champion at the final per-class decode (E375: short 0.3 / long 0.2 / keep 0.6 / gate off), system champion_pc_s0, added to a COPY of pass1_with_champion.pkl; CPU only
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py graphs --run e373_cb_s0 --seed 0 --name champion_pc_s0 \
  --out temp/qual_report_1117b/pass1_with_champion.pkl --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate
echo "exit $?"
