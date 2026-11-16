#!/bin/bash
# E322 gate G0 with the promoted script evaluation/jwiring.py (the numbers the card cites): both seeds, CPU, nice
cd "$(dirname "$0")/../.."
for S in 0 1; do nice bash tools/env/_py14.sh jwiring.py score --run e312_p_s$S --seed $S >> temp/e322/g0p_e312_s$S.log 2>&1; done
nice bash tools/env/_py14.sh jwiring.py reach --run e312_p_s0 --seed 0 >> temp/e322/g0p_reach_s0.log 2>&1
nice bash tools/env/_py14.sh jwiring.py reach --run e312_p_s1 --seed 1 >> temp/e322/g0p_reach_s1.log 2>&1
echo done > temp/e322/g0p.done
