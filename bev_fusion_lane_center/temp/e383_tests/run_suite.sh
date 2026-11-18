#!/bin/bash
# System test, stream A: tools/maintenance/run_all_tests.sh (its GPU tests only from 22:00, Mirko's / the lead's rule) with its output in temp/e383_tests/suite.out.
cd "$(dirname "$0")/../.."
until [ "$(date +%H%M)" -ge 2200 ] && [ "$(date +%H%M)" -lt 2359 ]; do sleep 20; done
if pgrep -f "run.py .*--gpu" > /dev/null; then echo "a GPU run.py job is alive: not parallel to it" > temp/e383_tests/suite.out; exit 1; fi
echo "start $(date +%H:%M:%S)" > temp/e383_tests/suite.out
nice -n 5 bash tools/maintenance/run_all_tests.sh >> temp/e383_tests/suite.out 2>&1
echo "end $(date +%H:%M:%S) exit $?" >> temp/e383_tests/suite.out
