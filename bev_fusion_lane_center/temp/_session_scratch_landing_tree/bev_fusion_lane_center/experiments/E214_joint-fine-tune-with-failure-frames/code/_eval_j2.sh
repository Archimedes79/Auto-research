#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# evaluation of E214: route success (p17_route) and graph quality (p17_eval) on the VAL failure series with the spatial hold, M 192 / K0 96, no time recursion, route-head seeds 0 / 1
R=$BFLC
X=$R/experiment_execution/runs
for a in a0 c0 a1 c1; do until [ -f $X/E214.$a/done ]; do sleep 30; done; done
SER=dark_maxdecay_0.95,blur_maxdecay_0.95,cam_front3_off_maxdecay_0.95
SET="M=192 K0=96"
set -e
bash $R/tools/env/_pyr.sh p17_route.py --out e18_j2_route --set $SET --prev 0 --series $SER --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0 --tag_suffix _base
bash $R/tools/env/_pyr.sh p17_eval.py --out e18_j2 --set $SET --prev 0 --perts dark,blur,cam_front3_off --variants maxdecay:0.95 --suffix _base
for arm in a0 c0 a1 c1; do
  bash $R/tools/env/_pyr.sh p17_route.py --out e18_j2_route --set $SET --prev 0 --ckpt e18_joint_$arm --series $SER --routes e18_joint_$arm --sigmas 0 --tag_suffix _$arm
  bash $R/tools/env/_pyr.sh p17_eval.py --out e18_j2 --set $SET --prev 0 --ckpt e18_joint_$arm --perts dark,blur,cam_front3_off --variants maxdecay:0.95 --suffix _$arm
done
