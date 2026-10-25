#!/bin/bash
# E240 layer 1: the stored seeds 0 / 1 of j1rc and j1eq evaluated with the one entry point (topo_F1 and R@1_far on the cards), so decide.py --experiment E240 / E237 can read them
cd "$(dirname "$0")/../.."
for n in e24_j1rc_s0 e24_j1rc_s1 e24_j1eq_s0 e24_j1eq_s1; do
  echo "== $n"
  bash tools/env/_py14.sh evaluate.py $n 2>&1 | grep -E "^  S|on val" | cut -c1-230
done
