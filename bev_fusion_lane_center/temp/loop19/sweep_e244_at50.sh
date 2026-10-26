#!/bin/bash
# E244 partial reading: tau sweep of the 50 % checkpoint of arm a (cached raw outputs, CPU)
cd "$(dirname "$0")/../.."
for t in ${@:-0.6 0.7 0.8 0.9}; do
  echo "== tau $t"
  bash tools/env/_py14.sh evaluate.py e24_j1phd_s0_at50 --tau $t --tag tau$t 2>&1 | grep -E "^  S" | cut -c1-230
done
