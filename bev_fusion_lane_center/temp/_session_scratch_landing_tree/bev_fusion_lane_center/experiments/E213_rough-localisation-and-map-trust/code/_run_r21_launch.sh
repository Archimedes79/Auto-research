#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
R=$BFLC
python3 $R/tools/experiment/run.py --id E213.a --gpu-max-mb 9500 --retries 2 --basis E207,E206,E210 --seed 0 \
  --hyp "rough localisation (K145) + trust gate (K146) restore the map benefit at 1-3 m map error and switch off wrong maps; builder B unchanged" -- \
  bash $R/experiments/E213_rough-localisation-and-map-trust/code/_run_r21.sh > $R/runs/e17/logs/e213_a.log 2>&1
