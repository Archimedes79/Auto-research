#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# final numbers of the report: master, extra, figures, report html
S=$BFLC/training
L=$BFLC/runs/e17/logs
bash $BFLC/tools/env/_py14.sh p17_master.py > $L/final_master.log 2>&1
bash $BFLC/tools/env/_py14.sh p17_extra.py > $L/final_extra.log 2>&1
bash $BFLC/tools/env/_py14.sh fig_p17_bars.py > $L/final_figs.log 2>&1
bash $BFLC/tools/env/_py14.sh fig_p17_best.py >> $L/final_figs.log 2>&1
bash $BFLC/tools/env/_py14.sh build_loop17_report.py > $L/final_build.log 2>&1
echo finished > $L/final17.done
