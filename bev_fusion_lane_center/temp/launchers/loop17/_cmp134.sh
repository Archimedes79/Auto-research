#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E134: learned gate against the training-free spatial hold (same builder, recursion off) and against the champion with recursion + hold
S=$REPO/scripts
for p in dark blur cam_front3_off; do
  bash $S/_py14.sh p17_compare.py --pert $p --md tables/e134.md \
    --entries single=builder:e132_norec/${p}_single hold=builder:e132_norec/${p}_maxdecay_0.95 gate=builder:e134_gate/${p}_gate ctrl_md=control:maxdecay_0.95 \
    --boot "gate,hold;gate,single;gate,ctrl_md" 2>&1 | tail -13
done
