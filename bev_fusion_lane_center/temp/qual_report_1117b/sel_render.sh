#!/bin/bash
# scratch (2026-11-17): selection rules + render at the final per-class decode; CPU only
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2; PWD_R=$(pwd)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
true
nice -n 10 bash tools/env/_py14.sh render_pc.py $PWD_R/temp/qual_report_1117b/spec_champion_q1117.json $PWD_R/temp/qual_report_1117b/final
echo "exit $?"
