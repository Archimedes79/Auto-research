#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# evaluation of E232 on the VAL failure series (spatial hold), M 192 / K0 96, no time recursion, route seed 0: base head vs arm c vs arm a
R=$BFLC
X=$R/experiment_execution/runs
for a in a0 c0; do until [ -f $X/E232.$a/done ]; do sleep 30; done; done
SER=dark_maxdecay_0.95,blur_maxdecay_0.95,cam_front3_off_maxdecay_0.95
SET="M=192 K0=96"
set -e
for arm in a0 c0; do
  bash $R/tools/env/_pyr.sh p17_route.py --out e18_j3_route --set $SET --prev 0 --ckpt e18_joint2_$arm --series $SER --routes e18_joint2_$arm --sigmas 0 --tag_suffix _$arm
  bash $R/tools/env/_pyr.sh p17_eval.py --out e18_j3 --set $SET --prev 0 --ckpt e18_joint2_$arm --perts dark,blur,cam_front3_off --variants maxdecay:0.95 --suffix _$arm
done
