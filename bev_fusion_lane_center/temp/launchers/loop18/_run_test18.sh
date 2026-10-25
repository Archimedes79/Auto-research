#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# TEST (once), loop 18: graph metrics + route head along the edges (original heads and failure-trained heads) of f1 / j1 with fork pruning 0.3, and of the control. Started only after E219 / E220 have finished.
# usage: _run_test18.sh [rc <run name> <M> <K0> <fork_prune>]   (the recall variant is added when it has been selected on VAL)
R=$BFLC
L=$R/temp/launchers/loop18
X=$R/experiment_execution/runs
until [ -f $X/E220.a/done ] && { [ -f $X/E219.rc2/done ] || [ -f $X/E219.rc2/give_up ]; }; do sleep 30; done
RUN="python3 $R/tools/experiment/run.py --split test --gpu-max-mb 9000 --retries 1 --seed 0"
$RUN --id E230.f1p --basis E215,E214 --hyp "champion f1 + fork pruning keeps its VAL advantage on TEST" --pred "S >= 0.75, J >= 0.35, route success >= 0.80 (TEST)" --rule "report; nothing is selected on TEST" -- bash $L/_pipeline18b.sh test_clean test18 e123_D7_r3_f1 288 144 0.3 f1p
$RUN --id E230.j1p --basis E215 --hyp "single-frame j1 192 / 96 + fork pruning as the fast deployment setting" --pred "S >= 0.74, J >= 0.35" --rule "report; nothing is selected on TEST" -- bash $L/_pipeline18b.sh test_clean test18 e123_D7_r3_j1 192 96 0.3 j1p
$RUN --id E230.ctrl --basis E185 --hyp "control E94 with the route head along the edges on TEST" --pred "route success 0.85 - 0.88" --rule "reference only" -- bash $L/_test18_ctrl.sh
if [ "$1" = "rc" ]; then
  $RUN --id E230.rc --basis E219,E215 --hyp "recall variant $2 + fork pruning $5 on TEST" --pred "Q above the control, lanes found above 0.74" --rule "report; nothing is selected on TEST" -- bash $L/_pipeline18b.sh test_clean test18 $2 $3 $4 $5 rcp
fi
