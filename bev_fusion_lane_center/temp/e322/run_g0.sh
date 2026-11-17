#!/bin/bash
# E322 gate G0 (scratch): oracle selection of the wired piece, E312 keypoint ends, both seeds. CPU, nice.
cd "$(dirname "$0")/../.."
for S in 0 1; do nice bash tools/env/_py14.sh ../temp/e322/oracle_wiring.py e312_p_s$S $S >> temp/e322/g0_e312_s$S.log 2>&1; done
echo done > temp/e322/g0.done
