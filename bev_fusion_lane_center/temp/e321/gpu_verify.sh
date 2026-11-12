#!/bin/bash
# fork C scratch (not committed): the GPU check of the dumps on VAL, as diagnostic runs of the launcher (no card, never feeds a verdict); run it in a free GPU slot BEFORE the TEST run
cd "$(dirname "$0")/../.."
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E321.v_s$S --no-card --gpu --retries 0 --split val --seed $S \
    --hyp "the dumps of jtest_eval.py made on the GPU reproduce the stored VAL cards of E312 (check of the TEST pipeline, D84)" \
    --pred "all numbers equal to 1e-6" --rule "diagnostic: no verdict" \
    -- bash tools/env/_pyr.sh evaluation/jtest_eval.py verify --seed $S --fresh --device cuda --boot 0
done
