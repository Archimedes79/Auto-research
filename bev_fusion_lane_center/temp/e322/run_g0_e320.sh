#!/bin/bash
# E322 gate G0 on the attached ends of E320 (the rule at the attached end against the rule at the GT end), both seeds, CPU, nice
cd "$(dirname "$0")/../.."
for S in 0 1; do nice bash tools/env/_py14.sh jwiring.py score --run e320_a_s$S --seed $S --attach >> temp/e322/g0p_e320_s$S.log 2>&1; done
echo done > temp/e322/g0e320.done
