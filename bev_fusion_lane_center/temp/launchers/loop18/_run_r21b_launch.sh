#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
R=$BFLC
python3 $R/tools/experiment/run.py --id E213.b --gpu-max-mb 9500 --retries 2 --basis E213.a,E207 --seed 0 \
  --hyp "registration + trust gate also lift the best chain (map builder + learned gate) at 1-3 m map error and against wrong maps" -- \
  bash $R/temp/launchers/loop18/_run_r21b.sh > $R/runs/e17/logs/e213_b.log 2>&1
