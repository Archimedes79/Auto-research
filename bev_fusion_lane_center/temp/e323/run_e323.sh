#!/bin/bash
# E323: the three read-out scores per seed (CPU, nice 19, one stream per seed): E312 snapped and snap-nogeom at t1, E320 PRIMARY at t2 (card E323, section 0)
cd "$(dirname "$0")/../.."
run_seed() {
  S=$1; T1=$2; T2=$3
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --snap --thr $T1 >> temp/e323/score_c_s$S.log 2>&1
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --snap --snap-nogeom --thr $T1 >> temp/e323/score_u_s$S.log 2>&1
  nice -n 19 bash tools/env/_py14.sh jends_eval.py score --run e320_a_s$S --seed $S --attach --snap --snap-nogeom --thr $T2 >> temp/e323/score_a_s$S.log 2>&1
  echo "seed $S done $(date)" >> temp/e323/run_e323.log
}
run_seed 0 0.083 0.12 &
run_seed 1 0.09 0.11 &
wait
echo done > temp/e323/e323.done
