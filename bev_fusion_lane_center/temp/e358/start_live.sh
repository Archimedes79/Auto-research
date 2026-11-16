#!/bin/bash
# E358 live check on the GPU (the device of the stored dumps), a run.py --gpu job with its own ledger row; diagnostic precondition of the training runs, feeds no verdict (--no-card).
#   usage: wsl bash temp/e358/start_live.sh <seed> [dry]
cd "$(dirname "$0")/../.."
S=$1; MODE=$2
[ -n "$S" ] || { echo "usage: start_live.sh <seed> [dry]"; exit 2; }
AFTER=""; [ "$S" = "1" ] && AFTER="--after E358.live_s0"
LINE=(python3 ../autoresearch/tools/run.py --id E358.live_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --timeout 1200 --no-card --split val $AFTER
  -- bash tools/env/_pyr.sh jwire_check.py live --seed $S --pair-run e324_s_s$S --max 24 --workers 2 --device cuda --ends att)
if [ "$MODE" = "dry" ]; then printf '%q ' "${LINE[@]}"; echo; else exec "${LINE[@]}"; fi
