#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
bash $REPO/scripts/_py14.sh p17_dense_train.py > $REPO/out/loop2/e17/logs/e139_dense.log 2>&1
echo finished > $REPO/out/loop2/e17/logs/e139_dense.done
