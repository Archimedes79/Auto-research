#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
for S in 0 1; do
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --snap --wire twin --tag twin > temp/_session_scratch_s292/regress_twin_s$S.log 2>&1
  echo "seed $S exit $?" >> temp/_session_scratch_s292/regress_twin.log
done
