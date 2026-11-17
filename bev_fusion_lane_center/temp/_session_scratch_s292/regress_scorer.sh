#!/bin/bash
# the tagged path of the scorer must reproduce the card of record of E312 seed 0 (same dump, same base, same spec) and leave the card of record alone
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s0 --seed 0 --thr 0.1 --snap --full $R/temp/e276/predfull_e273_p_s0.pkl --base $R/runs/e273_p_s0/model.pt --spec close:1.5@0.5 --tag regress > temp/_session_scratch_s292/regress_scorer.log 2>&1
echo "exit $?" >> temp/_session_scratch_s292/regress_scorer.log
