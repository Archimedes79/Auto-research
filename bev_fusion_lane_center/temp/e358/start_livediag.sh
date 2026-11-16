#!/bin/bash
cd "$(dirname "$0")/../.."
S=$1
exec python3 ../autoresearch/tools/run.py --id E358.livediag_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --timeout 1200 --no-card --split val -- bash tools/env/_pyr.sh live_diag.py --seed $S --bs 4
