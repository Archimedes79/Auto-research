#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
S=$REPO/scripts
L=$REPO/out/loop2/e17/logs
bash $S/_py14.sh p17_merge.py --prefix e133b --md tables/e133b_rerun.md > $L/merge133b.log 2>&1
bash $S/_py14.sh p17_master.py > $L/master.log 2>&1
