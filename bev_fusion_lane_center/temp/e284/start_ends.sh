#!/bin/bash
# E284 diagnostic: the junction end head (phase 8) on the frozen B run of one seed, through the launcher (--no-card: a diagnostic never feeds a verdict).  usage: start_ends.sh <seed> [smoke]
cd "$(dirname "$0")/../.."
S=$1
if [ "$2" = "smoke" ]; then ID=E284.smoke_s$S; NAME=e284_smoke_s$S; EXTRA="--smoke"; else ID=E284.diag_s$S; NAME=e284_diag_s$S; EXTRA=""; fi
python3 ../autoresearch/tools/run.py --id $ID --seed $S --gpu --retries 0 --no-card --split val \
  -- bash tools/env/_pyr.sh qgraph_train.py --name $NAME --seed $S --phases 8 --resume-from runs/e273_p_s$S/model.pt --workers 3 $EXTRA
