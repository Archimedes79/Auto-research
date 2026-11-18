#!/bin/bash
# E359 launch gate (card section 0): 50 steps, a diagnostic run (--no-card, feeds no verdict). Reads the `clock:` line of runs/e359_gate/log.txt.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
NAME=${1:-e359_gate}
[ -e runs/$NAME ] && { echo "runs/$NAME exists"; exit 2; }
python3 ../autoresearch/tools/run.py --id E359.$NAME --seed 0 --gpu --retries 0 --no-card --split val --timeout 25 \
  -- bash tools/env/_pyr.sh jtok2_train.py --name $NAME --seed 0 --lattice 2.5 --steps 50 --bs 8 --lr 1e-3 --workers 4
echo "exit $?"
grep -h "clock:" runs/$NAME/log.txt | tail -n 1 | cut -c1-300
