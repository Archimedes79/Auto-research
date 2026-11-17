#!/bin/bash
cd "$(dirname "$0")/../.."
for S in 0 1; do nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e316_p_s$S --seed $S --snap --snap-nogeom --thr 0.1 > temp/e324/e316_nogeom_s$S.log 2>&1; done
echo done > temp/e324/e316_nogeom.done
