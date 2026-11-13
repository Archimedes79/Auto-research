#!/bin/bash
# after the TEST run: does the CURRENT code (E320 / critic fixes installed while E321.t ran) still reproduce the stored VAL cards of E312 exactly? VAL only, CPU, nice.
cd "$(dirname "$0")/../.."
for S in 0 1; do
  nice bash tools/env/_py14.sh evaluation/jtest_eval.py verify --seed $S >> temp/e321/reverify_val_s$S.log 2>&1
done
echo done > temp/e321/reverify_val.done
