#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# waits for the latency measurement (idle GPU needed), then the E139 route evaluation, then the master results
S=$REPO/scripts
L=$REPO/out/loop2/e17/logs
until [ -f $L/e137.done ]; do sleep 20; done
bash $S/_run_e139_route.sh
bash $S/_py14.sh p17_master.py > $L/master2.log 2>&1
echo finished > $L/master2.done
