#!/bin/bash
# E354 launch gate (card section 0): 50 steps of arm r on one GPU slot, alone; a diagnostic run (--no-card), it feeds no verdict.
# Reads: seconds per step (steps 11 - 50), peak GPU memory, gate, state_contrib from the `clock:` line of runs/e354_gate/train.log.
# usage: gate.sh [seq] [bs] [name]      default 3 4 e354_gate
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
K=${1:-3}; BS=${2:-4}; NAME=${3:-e354_gate}
[ -e runs/$NAME ] && { echo "runs/$NAME exists"; exit 2; }
python3 ../autoresearch/tools/run.py --id E354.$NAME --seed 0 --gpu --retries 0 --no-card --split val --timeout 25 \
  -- bash tools/env/_pyr.sh jrecur_train.py --name $NAME --seed 0 --maps $(pwd)/runs/e332_ma_s0/model.pt --e2e --scales --maps-w 1.0 --steps 50 --seq $K --bs $BS --state on --workers 4
