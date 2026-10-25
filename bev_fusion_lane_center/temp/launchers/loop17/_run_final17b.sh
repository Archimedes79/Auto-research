#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
S=$BFLC/training
L=$BFLC/runs/e17/logs
bash $BFLC/tools/env/_py14.sh p17_extra.py > $L/final_extra.log 2>&1
bash $BFLC/tools/env/_py14.sh build_loop17_report.py > $L/final_build.log 2>&1
echo finished > $L/final17b.done
