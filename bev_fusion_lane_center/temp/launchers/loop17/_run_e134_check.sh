#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E134 sanity: the untrained gate reproduces the spatial hold
S=$REPO/scripts
bash $S/_py14.sh p17_gate.py apply --init_only --perts dark --out gate0
bash $S/_py14.sh p17_gate_check.py
