#!/bin/bash
# E372 50-step gate (card v3.2, section 1; diagnostic, run.py --no-card): z, b, s4 on seed 0, 50 steps each, one after another on the GPU queue.
# Started 23:45 on the lead's word (Mirko lifted the experiment stop at ~23:45; slot 1 is E372's tonight). Replaces the disarmed gate372.sh (it waited for E378.x_s1).
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e372/chain372.log
say() { echo "$(date '+%m-%d %H:%M:%S') [gate372b] $*" >> $L; }
exec 8>/tmp/e372_gate.lock
flock -n 8 || { say "gate already runs"; exit 0; }
for ARM in z b s4; do
  case $ARM in z) X="--train-nodes 0";; b) X="--train-nodes 1 --live-inputs 0";; s4) X="--train-nodes 1 --live-inputs 1";; esac
  say "gate $ARM starts ($X)"
  python3 ../autoresearch/tools/run.py --id E372.gate_${ARM}_s0 --seed 0 --gpu --ram-mb 8000 --retries 0 --no-card --split val --timeout 30 -- bash tools/env/_pyr.sh jlive_train.py --name e372_gate_${ARM}_s0 --seed 0 --head-run e373_cb_s0 $X --steps 50 --bs 8 --lr 1e-4 --workers 4 >> temp/e372/gate_${ARM}_s0.out 2>&1
  say "gate $ARM exit $?"
done
say "gate done"
