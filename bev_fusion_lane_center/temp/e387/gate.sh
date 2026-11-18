#!/bin/bash
# E387 launch gate (card section 1): the REAL 8000-step recipe of z (--no-past), r (--block residual) and f (--block fused), seed 0, killed at the timeout after >= 200 steps (run.py --no-card diagnostics, never a verdict); z and r in the two slots at once, then f alone (its s/step and memory are the numbers of the timeout). Then gate_check.py compares the step-100 / 200 rows with the stored E368 rows.
# usage: bash temp/e387/gate.sh      log: temp/e387/chain.log
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e387/texts387.sh
L=temp/e387/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [e387 gate] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
COMMON="--seed 0 --ident e292_a --steps 8000 --bs 4 --seq 3 --lr 1e-3 --workers 4"
say "start (z and r in the two slots)"
$RUN --id E387.gate_z_s0 --seed 0 --gpu --ram-mb 6000 --retries 0 --no-card --split val --timeout 5 -- bash tools/env/_pyr.sh jpast_train.py --name e387_gate_z_s0 $COMMON --no-past >> temp/e387/gate.out 2>&1 &
$RUN --id E387.gate_r_s0 --seed 0 --gpu --ram-mb 6000 --retries 0 --no-card --split val --timeout 5 -- bash tools/env/_pyr.sh jpast_train.py --name e387_gate_r_s0 $COMMON --block residual >> temp/e387/gate.out 2>&1 &
wait
say "z and r done; f alone"
$RUN --id E387.gate_f_s0 --seed 0 --gpu --ram-mb 6000 --retries 0 --no-card --split val --timeout 5 -- bash tools/env/_pyr.sh jpast_train.py --name e387_gate_f_s0 $COMMON --block fused >> temp/e387/gate.out 2>&1
say "f done"
python3 temp/e387/gate_check.py >> temp/e387/gate.out 2>&1
say "gate_check exit $?"
