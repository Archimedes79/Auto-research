#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E231.a after the two equal-budget controls are trained and evaluated: failure series of the j1 family through run.py (ledger row, done flag)
R=$BFLC
until [ -f $R/runs/e18/rc_eval/f1eq.json ] && [ -f $R/runs/e18/rc_eval/j1eqs1.json ]; do sleep 20; done
sleep 10
python3 $R/tools/experiment/run.py --id E231.a --gpu-max-mb 9000 --retries 1 --basis E219,E218 --seed 0 \
  --hyp "the recall weight also helps under sensor failure: j1rc finds more lanes than the equal-budget control j1eq on the night / blur / three-cameras-off series" \
  --pred "failure-half lanes found of j1rc - j1eq >= +0.03 in at least two of three failure types, Q >= +0.01" \
  --rule "lanes found >= +0.03 in >= 2 types AND Q not below j1eq - 0.005: the recall variant is also the robust choice; otherwise the recall gain is a clean-frame effect (report only, no new finalist)" -- \
  bash $R/experiments/E231_recall-variant-under-sensor-failure/code/_run_j1fam_fail.sh
