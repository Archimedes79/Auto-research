#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
L=$BFLC/temp/launchers/loop18
bash $L/_run_rc.sh rc2 6.0 &
sleep 90
bash $L/_run_rc.sh rc3 10.0 &
wait
