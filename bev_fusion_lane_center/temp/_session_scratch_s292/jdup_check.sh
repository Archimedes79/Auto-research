#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice bash tools/env/_py14.sh jdup.py selftest > temp/_session_scratch_s292/jdup_check.log 2>&1; echo "selftest exit $?" >> temp/_session_scratch_s292/jdup_check.log
nice bash tools/env/_py14.sh jdup.py score --seed 0 --a-run e320_a_s0 --c-run e312_p_s0 --l-run e316_p_s0 --max 150 >> temp/_session_scratch_s292/jdup_check.log 2>&1; echo "score exit $?" >> temp/_session_scratch_s292/jdup_check.log
