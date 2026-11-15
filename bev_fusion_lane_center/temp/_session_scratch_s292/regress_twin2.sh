#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s0 --seed 0 --thr 0.1 --snap --wire twin --tag twintwo > temp/_session_scratch_s292/regress_twin2.log 2>&1
echo "exit $?" >> temp/_session_scratch_s292/regress_twin2.log
