#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
nice -n 10 python3 ../autoresearch/tools/selftest.py > temp/e383_tests/selftest_alone.out 2>&1
echo "exit $?" >> temp/e383_tests/selftest_alone.out
