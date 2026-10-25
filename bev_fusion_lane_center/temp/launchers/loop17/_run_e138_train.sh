#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E138: map-conditioned builder, training (6 u from the champion, random map situation + perception loss on purpose)
S=$REPO/scripts
L=$REPO/out/loop2/e17/logs
bash $S/_py14.sh p17_mapft.py --name e17_map_a --u 6 --lr 5e-4 > $L/e138_train.log 2>&1
echo finished > $L/e138_train.done
