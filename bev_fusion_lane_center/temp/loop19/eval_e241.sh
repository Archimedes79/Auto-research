#!/bin/bash
# E241: evaluate the intermediate checkpoints (25 / 50 / 75 %) of the E240 fine-tunes of the given seeds on VAL 873 (evaluate.py, registry names from e241_prep.py)
#   bash temp/loop19/eval_e241.sh 2 [3]
cd "$(dirname "$0")/../.."
for seed in "$@"; do
  for p in 25 50 75; do
    for arm in j1rc j1eq; do
      n=e24_${arm}_s${seed}_at${p}
      echo "== $n"
      bash tools/env/_py14.sh evaluate.py $n 2>&1 | grep -E "^  S|on val" | cut -c1-230
    done
  done
done
