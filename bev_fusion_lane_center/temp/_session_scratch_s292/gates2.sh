#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice bash tools/maintenance/run_gates.sh > temp/_session_scratch_s292/gates2.log 2>&1; echo "exit $?" >> temp/_session_scratch_s292/gates2.log
