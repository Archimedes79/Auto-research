#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
S=$BFLC/training
L=$BFLC/runs/e17/logs
until [ -f $L/e140_route.done ]; do sleep 15; done
bash $BFLC/tools/env/_py14.sh p17_master.py > $L/master3.log 2>&1
bash $BFLC/tools/env/_py14.sh fig_p17_bars.py >> $L/master3.log 2>&1
bash $BFLC/tools/env/_py14.sh fig_p17_best.py >> $L/master3.log 2>&1
echo finished > $L/master3.done
