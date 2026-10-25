#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# R2.1c / E213.c: does the trust gate cost anything when the map is exact (sigma 0)?  Registration + gate (0.62 and 0.75) on the exact map, builder B (hold series) and builder B + gate (gateB series); references _m0 (exact, ungated) and _mnone
R=$BFLC
RG=$R/runs/e18/regis_eval_a
set -e
bash $R/tools/env/_pyr.sh p18_regis_eval.py --regis regis_a --out regis_eval_a --tags dark_maxdecay_0.95,blur_maxdecay_0.95,cam_front3_off_maxdecay_0.95,dark_gateB,blur_gateB,cam_front3_off_gateB --sigmas 0 --wrong 0
for thr in 0.62 0.75; do
  t=${thr/./}
  bash $R/tools/env/_pyr.sh p17_eval.py --out e18_r21c --perts dark,blur,cam_front3_off --mapnet 1 --ckpt e17_map_b --variants maxdecay:0.95 --map_mode exact --map_sigma 0 --reg $RG --reg_trust hard --reg_thr $thr --suffix _R${t}_s0
  bash $R/tools/env/_pyr.sh p17_eval.py --out e18_r21c --perts dark,blur,cam_front3_off --mapnet 1 --ckpt e17_map_b --variants gateB --map_mode exact --map_sigma 0 --reg $RG --reg_trust hard --reg_thr $thr --suffix _R${t}_s0
done
