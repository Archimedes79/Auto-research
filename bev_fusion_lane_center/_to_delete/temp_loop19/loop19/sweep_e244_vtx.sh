#!/bin/bash
# E244 partial reading: vertex-radius sweep at tau 0.12 of the 50 % checkpoint of arm a (match the 14.3 vertex budget; cached raw outputs, CPU)
cd "$(dirname "$0")/../.."
for v in "$@"; do
  echo "== tau 0.12 vtx_r $v"
  bash tools/env/_py14.sh evaluate.py e24_j1phd_s0_at50 --tau 0.12 --asm vtx_r=$v --tag t12v$v 2>&1 | grep -E "^  S" | cut -c1-230
done
