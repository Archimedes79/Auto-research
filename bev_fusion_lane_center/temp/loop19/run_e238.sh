#!/bin/bash
# E238: registration + trust against no registration on two NEW draws of the scene-constant 2 m map error (P19_MAP_SEED = 1, 2; the stored draw is 0 = E213.b),
# failure series dark / blur / cam_front3_off, chain "gateB" (map builder B + learned gate, checkpoint e17_map_b), registration regis_a, hard trust at 0.62.
# Commands as E213.b (experiments/E213_*/code/_run_r21b.sh). Output series: runs/e17/e19_e238/<pert>_gateB_{R62,U2}_d<k>; comparison: temp/loop19/cmp_e238.sh
cd "$(dirname "$0")/../.."
set -e
P="bash tools/env/_pyr.sh"
T=dark_gateB,blur_gateB,cam_front3_off_gateB
for k in 1 2; do
  export P19_MAP_SEED=$k
  echo "== draw $k: registration (regis_a, sigma 2)"
  $P p18_regis_eval.py --regis regis_a --out regis_eval_a_d$k --tags $T --sigmas 2 --wrong 0
  E="$P p17_eval.py --out e19_e238 --perts dark,blur,cam_front3_off --mapnet 1 --ckpt e17_map_b --variants gateB"
  echo "== draw $k: unregistered U2"
  $E --map_mode exact --map_sigma 2 --suffix _U2_d$k
  echo "== draw $k: registered R2 (hard trust 0.62)"
  $E --map_mode exact --map_sigma 2 --reg $PWD/runs/e18/regis_eval_a_d$k --reg_trust hard --reg_thr 0.62 --suffix _R62_d$k
done
echo "E238 evaluations done"
